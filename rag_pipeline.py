import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pypdf import PdfReader
from docx import Document as DocxDocument

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()


def parse_company_year_from_filename(file_path):
    """Derives company/year from a Company_Year filename convention,
    e.g. 'NVIDIA_2025.pdf' -> ('NVIDIA', '2025')."""
    stem = Path(file_path).stem
    parts = stem.replace("-", "_").split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        return "_".join(parts[:-1]), parts[-1]
    return stem, "N/A"

# -----------------------------
# Load PDF file
# -----------------------------
def load_pdf(file_path):
    docs = []
    reader = PdfReader(file_path)

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text and text.strip():
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(file_path),
                        "page": page_number
                    }
                )
            )

    return docs


# -----------------------------
# Load DOCX file
# -----------------------------
def load_docx(file_path):
    doc = DocxDocument(file_path)
    text_parts = []

    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            text_parts.append(paragraph.text.strip())

    for table in doc.tables:
        for row in table.rows:
            row_text = []
            for cell in row.cells:
                if cell.text.strip():
                    row_text.append(cell.text.strip())

            if row_text:
                text_parts.append(" | ".join(row_text))

    text = "\n".join(text_parts)

    if not text.strip():
        return []

    return [
        Document(
            page_content=text,
            metadata={"source": str(file_path)}
        )
    ]


# -----------------------------
# Load CSV file
# -----------------------------
def load_csv(file_path):
    df = pd.read_csv(file_path)

    if df.empty:
        return []

    text = f"""
CSV File: {Path(file_path).name}

Total Rows: {len(df)}
Total Columns: {len(df.columns)}

Columns:
{", ".join(df.columns.astype(str))}

Data Preview:
{df.head(100).to_string(index=False)}
"""

    return [
        Document(
            page_content=text,
            metadata={"source": str(file_path)}
        )
    ]


# -----------------------------
# Load Excel file
# -----------------------------
def load_excel(file_path):
    docs = []
    excel_file = pd.ExcelFile(file_path)

    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet_name)

        if df.empty:
            continue

        text = f"""
Excel File: {Path(file_path).name}
Sheet Name: {sheet_name}

Total Rows: {len(df)}
Total Columns: {len(df.columns)}

Columns:
{", ".join(df.columns.astype(str))}

Data Preview:
{df.head(100).to_string(index=False)}
"""

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(file_path),
                    "sheet": sheet_name
                }
            )
        )

    return docs


# -----------------------------
# Load TXT file
# -----------------------------
def load_txt(file_path):
    text = Path(file_path).read_text(encoding="utf-8", errors="ignore")

    if not text.strip():
        return []

    return [
        Document(
            page_content=text,
            metadata={"source": str(file_path)}
        )
    ]


# -----------------------------
# Load single file
# -----------------------------
def load_single_file(file_path):
    """
    This function decides which loader to use based on the uploaded file
    extension, then enriches every returned Document's metadata with
    company/year/document_type — parsed from the filename — so retrieval
    and citations can be company-specific, not just filename-based.
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        docs = load_pdf(file_path)
    elif suffix == ".docx":
        docs = load_docx(file_path)
    elif suffix == ".csv":
        docs = load_csv(file_path)
    elif suffix in [".xlsx", ".xls"]:
        docs = load_excel(file_path)
    elif suffix == ".txt":
        docs = load_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    company, year = parse_company_year_from_filename(file_path)
    for doc in docs:
        doc.metadata.setdefault("company", company)
        doc.metadata.setdefault("year", year)
        doc.metadata.setdefault("document_type", "Annual Report")

    return docs


# -----------------------------
# Format retrieved documents
# -----------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# -----------------------------
# Build RAG pipeline
# -----------------------------
def build_rag(file_paths):
    """
    Main RAG function.

    Steps:
    1. Load uploaded files
    2. Extract text
    3. Split into chunks
    4. Create embeddings
    5. Store in FAISS
    6. Create retriever
    7. Connect retriever with Groq LLM
    """

    start_time = time.time()

    all_documents = []

    for file_path in file_paths:
        docs = load_single_file(file_path)
        all_documents.extend(docs)

    print("Documents loaded:", len(all_documents))

    if not all_documents:
        raise ValueError(
            "No text was extracted from uploaded files. "
            "Try a text-based PDF, DOCX, CSV, XLSX, or TXT file. "
            "Scanned PDFs/images need OCR."
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150
    )

    chunks = text_splitter.split_documents(all_documents)

    print("Chunks created:", len(chunks))

    if not chunks:
        raise ValueError(
            "Documents were loaded, but no chunks were created. "
            "Please check your uploaded file content."
        )

    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name
    )

    vector_store = FAISS.from_documents(chunks, embeddings)

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 8}
    )

    # Groq deprecated llama-3.1-8b-instant on 2026-06-17; openai/gpt-oss-120b
    # is their recommended, currently-supported replacement.
    llm_model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    llm = ChatGroq(
        model=llm_model_name,
        temperature=0,
        max_tokens=1500 
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful RAG assistant.

Answer the question using only the provided context.

If the answer is not available in the context, say:
"I could not find this information in the uploaded documents."

Context:
{context}

Question:
{question}

Answer:
"""
    )

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )

    end_time = time.time()

    stats = {
        "total_files": len(file_paths),
        "total_documents": len(all_documents),
        "total_chunks": len(chunks),
        "embedding_model": embedding_model_name,
        "vector_database": "FAISS",
        "llm_model": llm_model_name,
        "processing_time": round(end_time - start_time, 2),
        "status": "Ready"
    }


    # NEW: vector_store and llm are returned so metric extraction (Phase 2)
    # and company-filtered retrieval (Phase 5) can reuse them directly,
    # instead of re-building embeddings or re-connecting to Groq.
    return rag_chain, stats, retriever, vector_store, llm

# =============================================================================
# AGENTIC LAYER — wraps the existing RAG retriever + financial_tools.py
# functions as LangChain tools, and lets the LLM decide which to call.
# =============================================================================
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate as AgentPromptTemplate

from financial_tools import get_financial_data, calculate_metric, calculate_margin


def build_agent(retriever, llm):
    """
    Builds a tool-calling agent that chooses between:
      - search_documents  (qualitative RAG — risks, strategy, management commentary)
      - get_company_financial_data  (a single numeric lookup)
      - calculate_financial_metric  (growth rate / CAGR / difference across two years)
      - calculate_financial_margin  (ratio between two metrics in the same year)

    This is the "Research Agent" in the architecture diagram — the retriever,
    financial data layer, and calculator all become tools it can call.
    """

    @tool
    def search_documents(query: str) -> str:
        """
        Use this for QUALITATIVE questions about a company — risks, strategy,
        management commentary, growth drivers, AI investments, competitive
        landscape. Do NOT use this for specific numeric values like revenue
        or net income — use get_company_financial_data for those instead.
        """
        docs = retriever.invoke(query)
        return format_docs(docs)

    @tool
    def get_company_financial_data(company: str, metric: str, year: int) -> str:
        """
        Use this for a SINGLE numeric financial figure — e.g. "What was
        NVIDIA's revenue in 2025?" or "What was Apple's net income in 2024?"
        metric examples: "Revenue", "Total net sales", "Net income",
        "Total assets", "Total liabilities", "Gross margin".
        Returns the exact figure with its unit, fiscal year, and source file.
        """
        result = get_financial_data(company, metric, year)
        return str(result)

    @tool
    def calculate_financial_metric(operation: str, company: str, metric: str, year_a: int, year_b: int) -> str:
        """
        Use this for a CALCULATION across two years of the SAME metric —
        e.g. "Calculate NVIDIA's revenue growth from 2024 to 2025" or
        "What was Apple's revenue CAGR?".
        operation must be one of: "growth_rate", "difference", "cagr".
        year_a = earlier year, year_b = later year.
        """
        result = calculate_metric(operation, company, metric, year_a, year_b)
        return str(result)

    @tool
    def calculate_financial_margin(company: str, numerator_metric: str, denominator_metric: str, year: int) -> str:
        """
        Use this for a RATIO between two DIFFERENT metrics in the SAME year —
        e.g. "What was Apple's net profit margin in 2025?" means
        numerator_metric="Net income", denominator_metric="Total net sales".
        """
        result = calculate_margin(company, numerator_metric, denominator_metric, year)
        return str(result)

    tools = [
        search_documents,
        get_company_financial_data,
        calculate_financial_metric,
        calculate_financial_margin,
    ]

    agent_prompt = AgentPromptTemplate.from_messages([
        ("system", """You are an equity research assistant. Use only information
retrieved from the tools available to you. Do not invent financial figures.
If the requested information is unavailable, clearly state that it was not
found in the available sources.

For any financial value, always state the company, fiscal year, currency,
and unit before using it in your answer. Cite the source document/page when
available.

Choose tools carefully:
- Qualitative questions (risks, strategy, "why", management commentary) -> search_documents
- A single numeric figure -> get_company_financial_data
- Growth/CAGR/difference across two years of the same metric -> calculate_financial_metric
- A ratio between two different metrics in the same year (e.g. margin) -> calculate_financial_margin
- Comparisons across companies -> call the relevant tool once per company, then compare in your answer
"""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, agent_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return agent_executor