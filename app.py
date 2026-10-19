import os
import re
import shutil
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from langchain_core.documents import Document as LCDocument

from rag_pipeline import build_rag, format_docs, build_agent

load_dotenv()

# -----------------------------
# Global state
# -----------------------------
qa_chain = None
retriever = None
vector_store = None
llm = None
agent_executor = None          # the agent that routes between RAG + financial tools
rag_stats = {}
indexed_file_paths = []
indexed_reports = []          # [{"Company": ..., "Year": ..., "Filename": ...}, ...]
known_companies = []          # ["Apple", "Tesla", ...]
company_years = {}            # {"Apple": ["2024", "2025"], ...}
all_years = []                # ["2024", "2025", ...]

_last_used_docs = []          # populated by the search_documents tool each call,
                               # so ask_question() can still cite sources after
                               # routing through the agent.

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

COMPARE_KEYWORDS = ["compare", "versus", " vs ", "which company", "better", "outperform", "against each other"]

# Questions touching on these topics get an extra, targeted retrieval pass
# per company (similar to core financial-statement lookups) so that hard
# numbers like revenue/net income/EPS aren't missed just because the
# semantic search for the literal question wording happened to surface
# other pages instead (e.g. risk factors, depreciation notes, etc).
FINANCIAL_TRIGGER_KEYWORDS = [
    "compare", "performance", "revenue", "profit", "net income", "financial",
    "earnings", "eps", "margin", "growth", "outperform", "better", "results"
]
FINANCIAL_SUPPLEMENT_QUERY = (
    "Consolidated Statements of Operations. Total net sales. Total revenues. "
    "Net income for the fiscal year. Diluted earnings per share."
)

# Questions using these words get a longer, more detailed answer instead
# of the default terse summary.
DETAIL_KEYWORDS = [
    "detail", "details", "elaborate", "in depth", "in-depth", "thorough",
    "comprehensive", "deep dive", "explain", "walk me through", "fully"
]


# -----------------------------
# Helpers
# -----------------------------
def normalize_uploaded_files(file_paths):
    if file_paths is None:
        return []
    if not isinstance(file_paths, list):
        file_paths = [file_paths]

    normalized_paths = []
    for file in file_paths:
        if file is None:
            continue
        if isinstance(file, str):
            source_path = Path(file)
        elif hasattr(file, "name"):
            source_path = Path(file.name)
        elif isinstance(file, dict):
            source_path = Path(file.get("name") or file.get("path"))
        else:
            continue
        if source_path.exists():
            normalized_paths.append(source_path)
    return normalized_paths


def parse_report_name(report_stem):
    """
    'Apple_2024' -> ('Apple', '2024'). Falls back gracefully if a file
    isn't named in the Company_Year convention.
    """
    parts = report_stem.replace("-", "_").split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        return "_".join(parts[:-1]), parts[-1]
    return report_stem, "N/A"


def source_stem(doc):
    return Path(doc.metadata.get("source", "")).stem.lower()


def format_sources_inline(used_docs, max_sources=6):
    if not used_docs:
        return ""
    seen = []
    for d in used_docs:
        company = d.metadata.get("company")
        year = d.metadata.get("year")
        doc_type = d.metadata.get("document_type", "Annual Report")
        page = d.metadata.get("page")

        if company and company != "unknown":
            label = f"{company} {doc_type}"
            if year and year != "N/A":
                label += f" {year}"
            if page:
                label += f", p. {page}"
        else:
            name = Path(d.metadata.get("source", "unknown")).name
            label = f"{name} (p. {page})" if page else name

        if label not in seen:
            seen.append(label)
        if len(seen) >= max_sources:
            break
    return "\n\nSources: " + ", ".join(seen)


# -----------------------------
# Answer text cleanup — no markdown asterisks, no em dashes, no raw
# tool-name citation brackets. Applied as a final pass so the agent's
# own phrasing/logic never has to change, only the presentation.
# -----------------------------
TOOL_CITATION_LABELS = {
    "search_documents": "Document Search",
    "get_company_financial_data": "Financial Data",
    "calculate_financial_metric": "Calculated Metric",
    "calculate_financial_margin": "Calculated Margin",
}


def clean_answer_text(text):
    if not text:
        return text

    # Replace raw tool-name citation brackets, e.g. "\u3010calculate_financial_metric\u3011"
    def _replace_tool_citation(match):
        tool_name = match.group(1).strip()
        label = TOOL_CITATION_LABELS.get(tool_name, tool_name.replace("_", " ").title())
        return f"(Source: {label})"

    text = re.sub(r"\u3010([^\u3011]+)\u3011", _replace_tool_citation, text)

    # Strip markdown bold/italic markers, keep the inner text
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*(?!\*)([^\*\n]+?)\*(?!\*)", r"\1", text)

    # Replace em dashes with a comma — reads naturally in most sentence positions
    text = text.replace(" — ", ", ").replace("—", ",")

    # Convert markdown bullet markers ("* " / "- " at line start) into a
    # plain round bullet, which reads cleanly without being markdown syntax
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("* ") or stripped.startswith("- "):
            cleaned_lines.append("\u2022 " + stripped[2:].strip())
        else:
            cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)

    # Collapse any leftover double spaces created by the replacements above
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


# -----------------------------
# Smart, company-aware retrieval
# -----------------------------
def detect_mentioned_companies(question):
    q = question.lower()
    return [c for c in known_companies if c.lower() in q]


def detect_mentioned_years(question):
    return re.findall(r"\b(?:19|20)\d{2}\b", question)


def smart_retrieve(question, k_per_company=4):
    """
    Figures out which companies (and optionally years) a question is
    about, and retrieves a BALANCED set of chunks across all of them —
    instead of one global top-k search, which tends to starve out
    whichever company is less textually similar to the question and
    silently drop it from the answer.

    k_per_company lowered from 6 to 4, and the financial-supplement cap
    below from 4 to 2, to reduce per-call token usage — comparison
    questions across 2+ companies were the heaviest consumers against
    the daily Groq token quota.

    Returns (used_docs, formatted_context_string_with_company_sections).
    """
    mentioned = detect_mentioned_companies(question)
    q_lower = question.lower()
    is_comparison_like = any(kw in q_lower for kw in COMPARE_KEYWORDS)
    needs_financial_supplement = any(kw in q_lower for kw in FINANCIAL_TRIGGER_KEYWORDS)

    # "Compare all companies" / "which one is better" with no names given
    # -> treat every indexed company as mentioned.
    if not mentioned and is_comparison_like and len(known_companies) > 1:
        mentioned = list(known_companies)

    mentioned_years = detect_mentioned_years(question)

    if not mentioned:
        # Generic question, no company detected — use the normal retriever.
        docs = retriever.invoke(question) if retriever is not None else []
        return docs, format_docs(docs)

    grouped_context_parts = []
    all_used_docs = []

    for company in mentioned:
        wide_pool = vector_store.similarity_search(question, k=40)
        company_docs = [d for d in wide_pool if company.lower() in source_stem(d)]

        # If a year was mentioned and at least one chunk for this company
        # matches that year, narrow to just those chunks.
        if mentioned_years:
            year_matched = [d for d in company_docs if any(y in source_stem(d) for y in mentioned_years)]
            if year_matched:
                company_docs = year_matched

        company_docs = company_docs[:k_per_company]

        # For performance/comparison-style questions, always also pull the
        # core financial-statement chunks for this company, regardless of
        # how the actual question was worded — this is what prevents
        # "Apple's revenue" from silently disappearing from a comparison
        # just because the question phrasing favored a different page.
        if needs_financial_supplement:
            financial_pool = vector_store.similarity_search(FINANCIAL_SUPPLEMENT_QUERY, k=20)
            financial_docs = [d for d in financial_pool if company.lower() in source_stem(d)]
            if mentioned_years:
                year_matched = [d for d in financial_docs if any(y in source_stem(d) for y in mentioned_years)]
                if year_matched:
                    financial_docs = year_matched
            financial_docs = financial_docs[:2]

            existing_keys = {d.page_content[:80] for d in company_docs}
            for d in financial_docs:
                if d.page_content[:80] not in existing_keys:
                    company_docs.append(d)
                    existing_keys.add(d.page_content[:80])

        all_used_docs.extend(company_docs)

        if company_docs:
            block = "\n\n".join(doc.page_content for doc in company_docs)
            grouped_context_parts.append(f"=== {company} ===\n{block}")
        else:
            grouped_context_parts.append(f"=== {company} ===\n(No matching content retrieved for this company.)")

    return all_used_docs, "\n\n".join(grouped_context_parts)


class SmartRetrieverAdapter:
    """
    Wraps smart_retrieve() so it can be handed to build_agent() as a
    drop-in replacement for a plain retriever. build_agent()'s
    search_documents tool calls retriever.invoke(query) and formats the
    result with format_docs() — by returning a single pseudo-Document
    whose page_content IS the already-grouped "=== Company ===" context
    string, the company-aware balancing/formatting logic is preserved
    exactly as before, just now reachable through the agent.
    """

    def invoke(self, query):
        global _last_used_docs
        used_docs, context_str = smart_retrieve(query)
        _last_used_docs = used_docs  # stashed so ask_question() can cite sources afterward
        return [LCDocument(page_content=context_str, metadata={})]


BASE_ANSWER_STYLE = """
Formatting rules for your answer:
- Start with a single one-line summary of the answer, in plain prose (no markdown bold).
- Follow with a short list of the specific supporting figures or points (numbers, dates, names). Use plain lines, not markdown asterisks.
- Do not use markdown bold markers, asterisks, or em dashes anywhere in your answer.
- If the context is organized into "=== CompanyName ===" sections, address each company that was asked about, even if one section says no content was retrieved for it — say so plainly rather than ignoring that company.
- Do not repeat the question back verbatim.
"""

CONCISE_ANSWER_STYLE = BASE_ANSWER_STYLE + """
- Keep it brief and to the point: no more than 6 supporting points, and the whole answer under 150 words.
"""

DETAILED_ANSWER_STYLE = BASE_ANSWER_STYLE + """
- The user explicitly asked for detail, so be thorough: use up to 10 supporting points, add brief explanatory
  context around the numbers (not just the raw figures), and you may go up to 300 words.
"""


def wants_detailed_answer(question):
    q_lower = question.lower()
    return any(kw in q_lower for kw in DETAIL_KEYWORDS)


def ask_question(question, history):
    global _last_used_docs
    history = history or []

    if agent_executor is None:
        warning = "Please index your annual reports first using the upload panel below."
        history = history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": warning},
        ]
        return history, ""

    if not question or not question.strip():
        return history, ""

    try:
        _last_used_docs = []  # reset — repopulated only if search_documents tool is called
        style_instructions = DETAILED_ANSWER_STYLE if wants_detailed_answer(question) else CONCISE_ANSWER_STYLE

        # Style instructions travel with the question itself, since the
        # agent's system prompt is fixed at build time (built once when
        # reports are indexed), not per-question.
        agent_input = f"{question}\n\n{style_instructions}"

        response = agent_executor.invoke({"input": agent_input})
        answer_text = response.get("output", "").strip()
        answer_text = clean_answer_text(answer_text)

        # Only qualitative (search_documents) calls populate _last_used_docs —
        # a pure numeric/calculation answer won't have RAG "sources" to cite,
        # which is correct: it's citing the structured financial_data.csv
        # instead, already stated inline by the tool's own output.
        if _last_used_docs:
            answer_text += format_sources_inline(_last_used_docs)

        history = history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer_text},
        ]
        return history, ""

    except Exception as e:
        error_msg = f"Error while generating answer: {str(e)}"
        print(f"[app] ask_question failed: {e}")
        history = history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": error_msg},
        ]
        return history, ""


def clear_chat():
    return [], ""


def clear_all():
    global indexed_file_paths, indexed_reports, known_companies, company_years, all_years
    global qa_chain, retriever, vector_store, llm, rag_stats, agent_executor

    qa_chain = None
    retriever = None
    vector_store = None
    llm = None
    agent_executor = None
    rag_stats = {}
    indexed_file_paths = []
    indexed_reports = []
    known_companies = []
    company_years = {}
    all_years = []

    return (
        "", create_dashboard({}), [], create_reports_html([]),
        gr.update(choices=[], value=None),
        gr.update(choices=[], value=None),
        gr.update(choices=["Any"], value="Any"),
    )


# -----------------------------
# Dashboard HTML — every value now has an explicit, consistent light
# color so nothing renders dark-on-dark. !important added throughout
# since Gradio's own theme CSS otherwise wins on specificity for
# elements rendered inside gr.HTML.
# -----------------------------
def create_dashboard(stats):
    if not stats:
        return """
        <div style="padding:40px; border-radius:18px; background:#ffffff;
                    border:1.5px dashed #d6dbf5; text-align:center;">
            <div style="font-size:34px; margin-bottom:10px;">&#128193;</div>
            <h2 style="margin:0 0 6px; color:#1e1b4b !important; font-size:18px; font-weight:700;">No reports indexed yet</h2>
            <p style="color:#6b7280 !important; margin:0; font-size:13.5px;">
                Upload one or more annual reports below to build the index.
            </p>
        </div>
        """

    total_files = stats.get("total_files", "N/A")
    total_chunks = stats.get("total_chunks", "N/A")
    embedding_model = stats.get("embedding_model", "N/A")
    vector_database = stats.get("vector_database", "FAISS")
    llm_model = stats.get("llm_model", os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))
    companies = stats.get("companies_indexed", "N/A")
    years = stats.get("years_covered", "N/A")

    return f"""
    <div style="padding:28px; border-radius:20px;
                background:linear-gradient(145deg,#1e1b4b 0%,#3730a3 60%,#4338ca 100%);
                box-shadow:0 18px 48px rgba(30,27,75,0.32);
                border:1px solid rgba(165,180,252,0.22);">

        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:6px;">
            <h2 style="margin:0; font-size:20px; font-weight:800; color:#f5f6ff !important;">
                Index Overview
            </h2>
            <span style="background:rgba(129,140,248,0.20); border:1px solid rgba(165,180,252,0.45);
                         color:#e0e4ff !important; padding:5px 14px; border-radius:999px; font-size:11.5px; font-weight:700;
                         letter-spacing:0.4px;">
                AGENT READY
            </span>
        </div>
        <p style="color:#cdd3ff !important; margin:0 0 18px; font-size:12.5px;">
            Snapshot of what is currently indexed and searchable by the research agent.
        </p>

        <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:10px;">
            <div style="padding:14px; border-radius:14px; background:rgba(255,255,255,0.08); border:1px solid rgba(199,210,254,0.22);">
                <p style="margin:0; font-size:11.5px; color:#c7d2fe !important;">Companies</p>
                <p style="margin:6px 0 0; font-size:15px; font-weight:800; color:#ffffff !important;">{companies}</p>
            </div>
            <div style="padding:14px; border-radius:14px; background:rgba(255,255,255,0.08); border:1px solid rgba(199,210,254,0.22);">
                <p style="margin:0; font-size:11.5px; color:#c7d2fe !important;">Reports</p>
                <p style="margin:6px 0 0; font-size:20px; font-weight:800; color:#ffffff !important;">{total_files}</p>
            </div>
            <div style="padding:14px; border-radius:14px; background:rgba(255,255,255,0.08); border:1px solid rgba(199,210,254,0.22);">
                <p style="margin:0; font-size:11.5px; color:#c7d2fe !important;">Years</p>
                <p style="margin:6px 0 0; font-size:15px; font-weight:800; color:#ffffff !important;">{years}</p>
            </div>
        </div>

        <div style="margin-top:10px; display:grid; grid-template-columns:repeat(4, 1fr); gap:10px;">
            <div style="padding:14px; border-radius:14px; background:rgba(255,255,255,0.05); border:1px solid rgba(199,210,254,0.18);">
                <p style="margin:0; font-size:11.5px; color:#c7d2fe !important;">Chunks Indexed</p>
                <p style="margin:6px 0 0; font-size:16px; font-weight:800; color:#ffffff !important;">{total_chunks}</p>
            </div>
            <div style="padding:14px; border-radius:14px; background:rgba(255,255,255,0.05); border:1px solid rgba(199,210,254,0.18);">
                <p style="margin:0; font-size:11.5px; color:#c7d2fe !important;">Embeddings</p>
                <p style="margin:6px 0 0; font-size:12px; font-weight:700; color:#ffffff !important; word-break:break-word;">{embedding_model}</p>
            </div>
            <div style="padding:14px; border-radius:14px; background:rgba(255,255,255,0.05); border:1px solid rgba(199,210,254,0.18);">
                <p style="margin:0; font-size:11.5px; color:#c7d2fe !important;">Vector DB</p>
                <p style="margin:6px 0 0; font-size:15px; font-weight:800; color:#ffffff !important;">{vector_database}</p>
            </div>
            <div style="padding:14px; border-radius:14px; background:rgba(255,255,255,0.05); border:1px solid rgba(199,210,254,0.18);">
                <p style="margin:0; font-size:11.5px; color:#c7d2fe !important;">Agent Model</p>
                <p style="margin:6px 0 0; font-size:12px; font-weight:700; color:#ffffff !important; word-break:break-word;">{llm_model}</p>
            </div>
        </div>

        <div style="margin-top:12px; padding:12px 16px; border-radius:12px;
                    background:rgba(74,222,128,0.12); border:1px solid rgba(74,222,128,0.30);
                    font-size:12px; color:#dcfce7 !important;">
            Financial calculation tools active. Growth rate, CAGR, and margin
            questions are answered from structured data, not text search.
        </div>
    </div>
    """


def create_reports_html(rows):
    if not rows:
        return """
        <div style="padding:24px; border-radius:14px; background:#ffffff;
                    border:1.5px dashed #d6dbf5; color:#6b7280 !important; text-align:center; font-size:13px;">
            No reports indexed yet.
        </div>
        """

    row_html = ""
    for r in rows:
        row_html += f"""
        <tr>
            <td style="padding:11px 18px; border-bottom:1px solid #eceffc; color:#1e1b4b !important; font-weight:600;">{r['Company']}</td>
            <td style="padding:11px 18px; border-bottom:1px solid #eceffc; color:#1e1b4b !important;">{r['Year']}</td>
            <td style="padding:11px 18px; border-bottom:1px solid #eceffc; color:#6b7280 !important;">{r['Filename']}</td>
        </tr>
        """

    return f"""
    <div style="border-radius:14px; overflow:hidden; border:1px solid #dfe3f7; box-shadow:0 6px 18px rgba(30,27,75,0.05);">
        <table style="width:100%; border-collapse:collapse; font-size:13px; background:#ffffff;">
            <thead>
                <tr style="background:linear-gradient(135deg,#1e1b4b,#3730a3);">
                    <th style="padding:12px 18px; text-align:left; color:#e0e4ff !important; font-size:11.5px; letter-spacing:0.5px;">COMPANY</th>
                    <th style="padding:12px 18px; text-align:left; color:#e0e4ff !important; font-size:11.5px; letter-spacing:0.5px;">YEAR</th>
                    <th style="padding:12px 18px; text-align:left; color:#e0e4ff !important; font-size:11.5px; letter-spacing:0.5px;">FILE</th>
                </tr>
            </thead>
            <tbody>
                {row_html}
            </tbody>
        </table>
    </div>
    """


def upload_files(file_paths):
    global qa_chain, retriever, vector_store, llm, rag_stats, agent_executor
    global indexed_file_paths, indexed_reports, known_companies, company_years, all_years

    uploaded_files = normalize_uploaded_files(file_paths)

    if not uploaded_files:
        return (
            "No valid files uploaded.", create_dashboard({}), create_reports_html([]), "",
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            gr.update(choices=["Any"], value="Any"),
        )

    saved_paths = []
    try:
        for source_path in uploaded_files:
            dest_path = DATA_DIR / source_path.name
            try:
                if source_path.resolve() != dest_path.resolve():
                    shutil.copy2(source_path, dest_path)
            except Exception:
                shutil.copy2(source_path, dest_path)
            saved_paths.append(str(dest_path))

        qa_chain, rag_stats, retriever, vector_store, llm = build_rag(saved_paths)
        indexed_file_paths = saved_paths

        # Build the agent, using our SmartRetrieverAdapter so the agent's
        # search_documents tool preserves the company-aware,
        # financial-supplement-aware retrieval logic from smart_retrieve().
        agent_executor = build_agent(SmartRetrieverAdapter(), llm)

                # Automatically extract structured financial data from any newly
        # uploaded PDF, so numeric lookups/calculations work for ANY
        # company — not just ones pre-processed through the notebook.
        # Failures here are logged but never block RAG indexing: a
        # document that isn't a standard financial statement (or where
        # extraction fails) still works fully for qualitative Q&A.
        from financial_extraction import extract_financial_data_from_pdf
        from financial_tools import append_financial_rows

        extraction_log = []
        for source_path in uploaded_files:
            if source_path.suffix.lower() != ".pdf":
                continue  # numeric extraction only applies to PDF financial statements

            company_name, _year = parse_report_name(source_path.stem)
            try:
                new_rows = extract_financial_data_from_pdf(str(DATA_DIR / source_path.name), company_name, llm=None)
                added = append_financial_rows(new_rows)
                if added > 0:
                    extraction_log.append(f"{company_name}: {added} financial figures extracted")
                else:
                    extraction_log.append(f"{company_name}: no new financial statement data found (RAG search still works)")
            except Exception as e:
                print(f"[app] Financial extraction failed for {source_path.name}: {e}")
                extraction_log.append(f"{company_name}: financial extraction failed, RAG search still works")

        parsed = [parse_report_name(Path(p).stem) for p in saved_paths]
        indexed_reports = [
            {"Company": c, "Year": y, "Filename": Path(p).name}
            for (c, y), p in zip(parsed, saved_paths)
        ]

        companies = sorted(set(c for c, _ in parsed))
        years = sorted(set(y for _, y in parsed if y != "N/A"))

        years_by_company = {}
        for c, y in parsed:
            years_by_company.setdefault(c, set()).add(y)
        years_by_company = {c: sorted(y_set) for c, y_set in years_by_company.items()}

        known_companies = companies
        company_years = years_by_company
        all_years = years

        rag_stats["companies_indexed"] = ", ".join(companies) if companies else "N/A"
        rag_stats["years_covered"] = ", ".join(years) if years else "N/A"

        reports_html = create_reports_html(indexed_reports)

        status_message = f"{len(saved_paths)} report(s) indexed. Agent ready. RAG and financial tools active.\n" + "\n".join(extraction_log)
        first_company = companies[0] if companies else None
        second_company = companies[1] if len(companies) > 1 else (companies[0] if companies else None)

        return (
            status_message,
            create_dashboard(rag_stats),
            reports_html,
            "",
            gr.update(choices=companies, value=first_company),
            gr.update(choices=companies, value=second_company),
            gr.update(choices=["Any"] + years, value="Any"),
        )

    except ValueError as e:
        return (
            f"{str(e)}", create_dashboard({}), create_reports_html([]), "",
            gr.update(choices=[], value=None), gr.update(choices=[], value=None),
            gr.update(choices=["Any"], value="Any"),
        )
    except Exception as e:
        return (
            f"Error: {str(e)}", create_dashboard({}), create_reports_html([]), "",
            gr.update(choices=[], value=None), gr.update(choices=[], value=None),
            gr.update(choices=["Any"], value="Any"),
        )


# -----------------------------
# Dynamic question builder
# -----------------------------
QUESTION_TEMPLATES = [
    "Compare two companies",
    "Compare profitability",
    "Why did revenue change",
    "What are the major risks",
    "Summarize financial performance",
    "Cash flow and liquidity",
    "Growth trends",
    "Management outlook and strategy",
    "Competitive position",
    "Cost and expense breakdown",
    "Calculate revenue growth",       # routes to calculate_financial_metric
    "Calculate profit margin",        # routes to calculate_financial_margin
]


def build_question(template, company_a, company_b, year):
    if not company_a:
        return "Please index reports and select a company first."

    year_suffix = "" if (not year or year == "Any") else f" in {year}"
    has_pair = bool(company_b and company_b != company_a)

    if template == "Compare two companies":
        if has_pair:
            return f"Compare the financial performance of {company_a} and {company_b}{year_suffix}."
        return f"Compare {company_a}'s financial performance across all indexed years."

    if template == "Compare profitability":
        if has_pair:
            return f"Compare the profitability, margins, and net income of {company_a} and {company_b}{year_suffix}."
        return f"How did {company_a}'s profitability and margins trend{year_suffix or ' over the indexed years'}?"

    if template == "Why did revenue change":
        return f"Why did {company_a}'s revenue change{year_suffix or ' year over year'}?"

    if template == "What are the major risks":
        return f"What are the major risks mentioned in {company_a}'s report{year_suffix}?"

    if template == "Summarize financial performance":
        return f"Summarize {company_a}'s financial performance{year_suffix}."

    if template == "Cash flow and liquidity":
        return f"How is {company_a}'s cash flow and liquidity position{year_suffix}?"

    if template == "Growth trends":
        return f"What are {company_a}'s key growth trends{year_suffix or ' across the indexed years'}?"

    if template == "Management outlook and strategy":
        return f"What does management say about future outlook or strategy{year_suffix} for {company_a}?"

    if template == "Competitive position":
        return f"What is {company_a}'s competitive position in its industry{year_suffix}?"

    if template == "Cost and expense breakdown":
        return f"Break down {company_a}'s major costs and operating expenses{year_suffix}."

    if template == "Calculate revenue growth":
        years_list = company_years.get(company_a, [])
        if len(years_list) >= 2:
            return f"Calculate {company_a}'s revenue growth from {years_list[0]} to {years_list[-1]}."
        return f"Calculate {company_a}'s revenue growth between its two most recent indexed years."

    if template == "Calculate profit margin":
        return f"What was {company_a}'s net profit margin{year_suffix or ' in the most recent indexed year'}?"

    return f"Tell me about {company_a}{year_suffix}."


# -----------------------------
# CSS — genuine redesign: refined type scale, card system, feature
# chips instead of a bullet list, consistent spacing, softer modern
# shadows. Indigo/violet accent kept but applied more deliberately.
# -----------------------------
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }

.gradio-container {
    max-width: 1180px !important;
    margin: auto !important;
    background: #eef0fb !important;
}

/* ---------- Hero ---------- */
#hero-section {
    padding: 32px 36px;
    border-radius: 20px;
    background: linear-gradient(145deg, #1e1b4b 0%, #322f7a 50%, #4338ca 100%);
    box-shadow: 0 20px 50px rgba(30,27,75,0.28);
    margin-bottom: 22px;
    border: 1px solid rgba(165,180,252,0.20);
}
#hero-section .eyebrow {
    display: inline-block;
    font-size: 11.5px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #c7d2fe !important;
    background: rgba(129,140,248,0.18);
    border: 1px solid rgba(165,180,252,0.35);
    padding: 4px 12px;
    border-radius: 999px;
    margin-bottom: 14px;
}
#hero-section h1 {
    font-size: 28px;
    margin: 0 0 10px;
    color: #ffffff !important;
    font-weight: 800;
    letter-spacing: -0.3px;
}
#hero-section .subtitle {
    font-size: 14px;
    color: #d7dbff !important;
    line-height: 1.65;
    margin: 0 0 20px;
    max-width: 720px;
}
#hero-section .feature-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-bottom: 16px;
}
#hero-section .feature-chip {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(199,210,254,0.20);
    border-radius: 12px;
    padding: 12px 14px;
    font-size: 12.5px;
    color: #e6e8ff !important;
    line-height: 1.5;
}
#hero-section .naming-tip {
    padding: 12px 16px;
    border-radius: 12px;
    background: rgba(129,140,248,0.14);
    border: 1px solid rgba(165,180,252,0.30);
    font-size: 12.5px;
    color: #eef0ff !important;
}
#hero-section code {
    background: rgba(0,0,0,0.25);
    padding: 2px 7px;
    border-radius: 6px;
    color: #b6c1ff !important;
    font-size: 12px;
}

/* ---------- Section labels ---------- */
.section-label {
    color: #1e1b4b !important;
    font-weight: 700 !important;
    font-size: 15.5px !important;
    margin-top: 26px !important;
    margin-bottom: 4px !important;
    display: flex;
    align-items: center;
    gap: 8px;
}

.footer-note { text-align: center; color: #9497b8; font-size: 11.5px; margin-top: 24px; }
.divider-line { border-top: 1px solid #dfe2f7; margin: 28px 0; }

/* ---------- Buttons ---------- */
.gradio-container button.primary {
    background: linear-gradient(135deg, #4f46e5, #4338ca) !important;
    border: none !important;
    box-shadow: 0 6px 16px rgba(79,70,229,0.32) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    font-weight: 600 !important;
    color: #ffffff !important;
}
.gradio-container button.primary:hover {
    transform: translateY(-1px);
    box-shadow: 0 10px 22px rgba(79,70,229,0.42) !important;
}
.gradio-container button:not(.primary) {
    border-radius: 10px !important;
    font-weight: 500 !important;
}

/* ---------- Chatbot ---------- */
.gradio-container .chatbot {
    border-radius: 16px !important;
    border: 1px solid #dfe2f7 !important;
    box-shadow: 0 6px 20px rgba(30,27,75,0.05) !important;
    background: #ffffff !important;
}

/* ---------- Inputs ---------- */
.gradio-container textarea, .gradio-container input[type="text"] {
    border-radius: 11px !important;
    border: 1.5px solid #dfe2f7 !important;
}
.gradio-container textarea:focus, .gradio-container input[type="text"]:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.14) !important;
}

/* ---------- Accordion ---------- */
.gradio-container .accordion {
    border-radius: 14px !important;
    border: 1px solid #dfe2f7 !important;
    background: #ffffff !important;
}
"""


# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks(title="FiscalLens") as demo:

    gr.HTML(
        """
        <div id="hero-section">
            <span class="eyebrow">Hybrid RAG • Agentic Financial Intelligence</span>
            <h1>FiscalLens</h1>
            <p class="subtitle">
                Turn complex annual reports into clear, reliable financial insights. Upload a company’s filings and ask questions in natural language, the intelligent agent determines whether to retrieve relevant information, look up an exact financial figure, or perform a financial calculation.
                From financial performance and business risks to growth trends and cross-company comparisons, every response is grounded in the available reports and structured data, not guesswork.
            </p>
            <div class="feature-grid">
                <div class="feature-chip">Every qualitative answer is grounded in the reports you uploaded, with the exact page cited.</div>
                <div class="feature-chip">Every number is retrieved or calculated from structured data extracted from the filing, never estimated by the model.</div>
                <div class="feature-chip">Ask about one company or compare several, across any fiscal year you've indexed.</div>
            </div>
            <div class="naming-tip">
                <strong>Hybrid RAG:</strong> FiscalLens combines document-based retrieval with structured
                financial data and intelligent calculation tools to provide grounded, data-driven answers.
                Naming convention: name your files <code>Company_Year</code>, for example
                <code>Tesla_2024.pdf</code> or <code>Apple_2025.pdf</code>. This lets the assistant
                automatically detect companies and years for filtering and comparisons.
            </div>
        </div>
        """
    )

    # ============================================================
    # SECTION 1 — Ask Questions
    # Question builder, then the input/Ask row directly below it,
    # then the conversation history — so asking a question never
    # requires scrolling past the chat panel first.
    # ============================================================
    gr.Markdown("### Ask Questions", elem_classes="section-label")

    with gr.Accordion("Question builder (optional, auto-fills based on your uploaded reports)", open=False):
        with gr.Row():
            template_dd = gr.Dropdown(choices=QUESTION_TEMPLATES, value=QUESTION_TEMPLATES[0], label="Question type")
            company_a_dd = gr.Dropdown(choices=[], label="Company")
            company_b_dd = gr.Dropdown(choices=[], label="Compare with (optional)")
            year_dd = gr.Dropdown(choices=["Any"], value="Any", label="Year")
        build_btn = gr.Button("Build question")

    with gr.Row():
        question = gr.Textbox(
            label="Your question",
            placeholder="e.g. Calculate NVIDIA's revenue growth from 2024 to 2025.",
            scale=4,
        )
        ask_btn = gr.Button("Ask", variant="primary", scale=1)

    clear_chat_btn = gr.Button("Clear conversation")



    chatbot = gr.Chatbot(
        label="Conversation",
        height=420,
        avatar_images=(None, "asset/robot_assistant.png"),
        elem_classes="chatbot",
    )

    gr.HTML('<div class="divider-line"></div>')

    # ============================================================
    # SECTION 2 — Upload + Index Overview
    # ============================================================
    gr.Markdown("### Upload Annual Reports", elem_classes="section-label")

    with gr.Row():
        with gr.Column(scale=1):
            file_upload = gr.File(
                label="Upload annual reports (name files as Company_Year, e.g. Tesla_2024.pdf)",
                file_types=[".pdf", ".docx", ".csv", ".xlsx", ".txt"],
                file_count="multiple",
                type="filepath"
            )
            upload_btn = gr.Button("Index reports", variant="primary", size="lg")
            clear_btn = gr.Button("Clear everything")
            status = gr.Textbox(label="Status", interactive=False, lines=3)

        with gr.Column(scale=2):
            dashboard = gr.HTML(value=create_dashboard({}))

    gr.Markdown("### Indexed Reports", elem_classes="section-label")
    reports_view = gr.HTML(value=create_reports_html([]))

    gr.HTML('<div class="footer-note">Built with LangChain agents, Groq, FAISS, Hugging Face Embeddings, and Gradio.</div>')

    # -----------------------------
    # Wiring
    # -----------------------------
    upload_btn.click(
        fn=upload_files,
        inputs=file_upload,
        outputs=[status, dashboard, reports_view, question, company_a_dd, company_b_dd, year_dd]
    )

    clear_btn.click(
        fn=clear_all,
        inputs=None,
        outputs=[status, dashboard, chatbot, reports_view, company_a_dd, company_b_dd, year_dd]
    )

    build_btn.click(
        fn=build_question,
        inputs=[template_dd, company_a_dd, company_b_dd, year_dd],
        outputs=question
    )

    ask_btn.click(fn=ask_question, inputs=[question, chatbot], outputs=[chatbot, question])
    question.submit(fn=ask_question, inputs=[question, chatbot], outputs=[chatbot, question])
    clear_chat_btn.click(fn=clear_chat, inputs=None, outputs=[chatbot, question])


demo.launch(debug=True, css=custom_css)