# FiscalLens

An agentic Retrieval-Augmented Generation (RAG) assistant for exploring
company annual reports. Built for the PGD Generative AI sessional project.

## 1. What this project does

The user uploads one or more company annual reports (PDF). The system:

- Indexes the reports for semantic search (RAG), so qualitative questions
  (risks, strategy, management commentary) are answered with grounded,
  page-cited excerpts from the actual documents.
- Automatically extracts structured financial data (revenue, net income,
  margins, balance sheet items, cash flow items) from each uploaded PDF,
  so exact numeric questions and calculations are answered from structured
  data rather than the LLM's own arithmetic or approximate text retrieval.
- Uses an agent that decides, per question, whether to search the
  documents, look up a single financial figure, or run a calculation
  (growth rate, CAGR, margin) — instead of a single fixed pipeline.

## 2. The Plan

The initial goal was a RAG chatbot answering questions from annual
reports. Two problems became clear early on:

1. **Plain text-based RAG cannot reliably answer numeric questions.**
   Financial statement tables lose their row/column structure when
   flattened into plain text by a standard PDF loader, risking a wrong
   number being associated with the wrong year.
2. A simple retrieval pipeline cannot decide "this needs a calculation"
   versus "this needs an explanation" — it always does the same thing.

The solution: a hybrid architecture combining unstructured RAG (for
qualitative content) with a structured financial dataset (for numeric
content), tied together by an agent that chooses the right tool per
question.

```
                        USER QUESTION
                              |
                       RESEARCH AGENT
                              |
        --------------------------------------------
        |                     |                     |
   TEXT QUERY            NUMERIC QUERY         CALCULATION QUERY
        |                     |                     |
      FAISS            financial_data.csv    growth / CAGR / margin
   (annual reports)     (structured figures)   (computed in Python,
                                                 never by the LLM)
        --------------------------------------------
                              |
                        Groq LLM (synthesis)
                              |
                      Answer + Sources
```

## 3. How the structured financial data is built

Extracting reliable numbers from financial statement PDFs took several
iterations to get right. The final, validated approach:

1. **Locate the right page** for each statement (Income Statement,
   Balance Sheet, Cash Flow Statement) using two independent signals
   together — specific accounting-term phrases (e.g. "Gross Margin",
   "Total Assets") AND the presence of real comma-formatted dollar
   figures (e.g. "130,497"). Requiring both prevents two failure modes
   found during testing: segment/geographic footnote tables (which reuse
   the same line-item names) and percentage-of-revenue analysis tables
   (which reuse the same vocabulary but only contain small decimals, not
   real dollar figures).
2. **Extract that page's text with layout preserved** (via `pdfplumber`),
   so columns stay visually aligned with their year headers.
3. **Send that text to an LLM with a strict JSON schema**, instructing it
   to only transcribe figures that are literally present — never
   calculate, estimate, or invent a number. Every row is tagged with a
   `section` (to disambiguate line items that repeat under different
   headings, e.g. "Products" appearing under both Net Sales and Cost of
   Sales) and a correctly-typed `unit` (dollar figures vs. per-share
   figures vs. raw share counts are never conflated).
4. **This now runs automatically** the first time a company's PDF is
   uploaded through the app — it does not require a separate manual
   step. Extraction failures degrade gracefully: if a page can't be
   confidently located (e.g. a non-standard report format), the document
   still works fully for qualitative RAG search; only the numeric lookup
   for that company would be unavailable.

This was validated end-to-end against real, published figures for
Apple, NVIDIA, and Tesla.

## 4. The Dataset

No dataset needs to be prepared in advance. `financial_data.csv` starts
empty (or with any previously processed companies) and grows automatically
as annual reports are uploaded through the app. Each row represents one
financial figure, tagged with company, statement type, section, metric
name, fiscal year, value, and unit, which allows exact lookups and
calculations without re-reading the source PDF each time.

Numeric extraction has been validated against real published figures for
Apple, NVIDIA, and Tesla. It targets standard US SEC-style filings
(10-Ks); reports using different terminology conventions may not be
automatically recognized for numeric extraction, though they remain
fully searchable as text.

Apple, NVIDIA, and Tesla are already present in the included dataset
file, so any of these companies can be uploaded immediately to test the
system without waiting for a first-time extraction pass.

## 5. The agent

The agent (built with LangChain's tool-calling agent) has four tools:

- `search_documents` — semantic search over the indexed reports, for
  qualitative questions.
- `get_company_financial_data` — a single exact numeric lookup from the
  structured dataset.
- `calculate_financial_metric` — growth rate, CAGR, or difference across
  two years of the same metric, computed in plain Python.
- `calculate_financial_margin` — a ratio between two different metrics
  in the same year (e.g. net profit margin), also computed in Python.

Calculations are always done in code, never left to the LLM to compute
from memory — this is the core guarantee against hallucinated numbers.

## 6. Known limitations (honest scope boundaries)

- Numeric extraction has been validated on US SEC-style filings (10-Ks).
  Reports using different terminology conventions (e.g. IFRS-format
  statements, such as "Statement of Financial Position" instead of
  "Consolidated Balance Sheet") are not currently recognized by the
  page-location logic, and numeric extraction may return no data for
  such a document. Qualitative RAG search still works regardless.
- Investor presentations / earnings slide decks have not been tested —
  they are typically formatted very differently from full financial
  statements and are out of scope for the current version.
- A small number of extracted figures should be spot-checked against
  the source PDF before being treated as fully authoritative in a
  presentation context, since LLM-based table transcription, while
  constrained and validated on multiple companies, is not infallible.
- Indexing a newly uploaded report can take one to a few minutes, since
  the system builds the FAISS search index and runs automatic financial
  data extraction (multiple language model calls to locate and
  transcribe statement pages) in the same step. This is a genuine
  processing cost, not a delay that can be skipped.

## 7. Project structure

```
app.py                   Gradio UI, question routing, upload handling
rag_pipeline.py           Document loading, chunking, embeddings, FAISS,
                          and the agent construction
financial_tools.py        Lookup and calculation functions used by the
                          agent (reads/writes financial_data.csv)
financial_extraction.py   Automatic structured-data extraction pipeline,
                          run on each newly uploaded PDF
financial_data.csv        The structured financial dataset (grows as
                          more companies are uploaded)
requirements.txt          Python dependencies
data/                     Uploaded report files are saved here
```

## 8. Setup and running

```
uv venv
uv pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

Run:
```
uv run app.py
```

Open the local URL shown in the terminal (typically `http://127.0.0.1:7860`).

## 9. User guide

1. **Upload reports.** Name each file `Company_Year.pdf` (e.g.
   `Apple_2025.pdf`, `Tesla_2024.pdf`) — this naming convention lets the
   system automatically detect the company and year for filtering,
   comparisons, and citations.
2. **Click "Index reports."** This builds the semantic search index and
   automatically attempts to extract structured financial figures from
   each PDF. This step can take one to a few minutes per file.
3. **Ask questions** either by typing directly, or using the question
   builder (select a question type, company, optional comparison
   company, and year) to auto-generate a well-formed question.
4. **Read the answer.** Qualitative answers cite the source document and
   page. Numeric answers state the company, fiscal year, and unit before
   the figure. Calculation answers show the underlying figures used.
5. To start over, use "Clear everything," which resets the index, the
   agent, and all indexed report metadata.

## 10. Example questions demonstrated during testing

- "What are the major risks mentioned in Apple's report?"
- "Calculate NVIDIA's revenue growth from 2024 to 2025."
- "Compare the financial performance of NVIDIA and Apple in 2025."
- "What does management say about future outlook or strategy for Tesla?"
- "How is Tesla's cash flow and liquidity position in 2024?"

## 11. Technology used

LangChain (agents, tool-calling), Groq (LLM inference), FAISS (vector
store), Hugging Face sentence-transformers (embeddings), pdfplumber
(layout-aware PDF table extraction), Gradio (web interface), pandas
(structured financial data).
