"""
Tool functions for the agentic layer. Same lookup/calculation logic as
before, but FINANCIAL_DATA is now reloadable and appendable at runtime,
so newly-extracted rows from a freshly-uploaded PDF can be added without
restarting the app.
"""

import os
import pandas as pd
import re

FINANCIAL_DATA_PATH = "financial_data.csv"

EXPECTED_COLUMNS = ["company", "statement", "section", "metric", "year", "value", "unit", "source_file"]


def _empty_financial_dataframe():
    return pd.DataFrame(columns=EXPECTED_COLUMNS)


def load_financial_data():
    """Loads financial_data.csv fresh from disk. Returns an empty (but
    correctly-shaped) DataFrame if the file doesn't exist yet — the app
    should still start even with zero pre-processed companies."""
    if os.path.exists(FINANCIAL_DATA_PATH):
        return pd.read_csv(FINANCIAL_DATA_PATH)
    return _empty_financial_dataframe()


# Loaded once at import for the common case (existing data already on disk),
# but every lookup function below re-reads via _get_data() so runtime
# appends are always reflected without needing to restart the app.
FINANCIAL_DATA = load_financial_data()


def _get_data():
    global FINANCIAL_DATA
    return FINANCIAL_DATA


def _normalize(s):
    if not isinstance(s, str):
        return s
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def append_financial_rows(new_rows):
    """
    Appends newly-extracted rows (from financial_extraction.py) to the
    in-memory dataset AND persists them to financial_data.csv, applying
    the same normalized deduplication used during the original batch
    extraction — so re-uploading the same company/year twice doesn't
    create duplicate rows.

    Returns the number of NEW rows actually added (after dedup), so the
    caller can report this to the user.
    """
    global FINANCIAL_DATA

    if not new_rows:
        return 0

    new_df = pd.DataFrame(new_rows)
    for col in EXPECTED_COLUMNS:
        if col not in new_df.columns:
            new_df[col] = None

    combined = pd.concat([_get_data(), new_df], ignore_index=True)

    combined["_company_norm"] = combined["company"].apply(_normalize)
    combined["_section_norm"] = combined["section"].apply(_normalize)
    combined["_metric_norm"] = combined["metric"].apply(_normalize)

    rows_before = len(_get_data())

    combined = combined.drop_duplicates(
        subset=["_company_norm", "statement", "_section_norm", "_metric_norm", "year"],
        keep="first"
    )
    combined = combined.drop(columns=["_company_norm", "_section_norm", "_metric_norm"])

    rows_after = len(combined)
    net_new = max(0, rows_after - rows_before)

    combined.to_csv(FINANCIAL_DATA_PATH, index=False)
    FINANCIAL_DATA = combined

    return net_new


def get_financial_data(company: str, metric: str, year: int) -> dict:
    """
    Looks up a single financial figure. Returns the value plus enough
    metadata (unit, section, source file) for the agent to cite it properly.
    """
    df = _get_data().copy()
    if df.empty:
        return {
            "found": False,
            "message": f"No financial data has been extracted yet for any company. "
                       f"Upload a PDF annual report to build the dataset."
        }

    df["_company_norm"] = df["company"].apply(_normalize)
    df["_metric_norm"] = df["metric"].apply(_normalize)

    company_norm = _normalize(company)
    metric_norm = _normalize(metric)

    exact_matches = df[
        (df["_company_norm"] == company_norm)
        & (df["_metric_norm"] == metric_norm)
        & (df["year"] == year)
    ]

    if not exact_matches.empty:
        matches = exact_matches
    else:
        matches = df[
            (df["_company_norm"] == company_norm)
            & (df["_metric_norm"].str.contains(metric_norm, na=False))
            & (df["year"] == year)
        ]

    if matches.empty:
        return {
            "found": False,
            "message": f"No data found for {company} — {metric} — {year}. "
                       f"If {company} hasn't been uploaded yet, upload its annual report PDF first."
        }

    unique_values = matches["value"].unique()
    row = matches.iloc[0]

    result = {
        "found": True,
        "company": row["company"],
        "metric": row["metric"],
        "year": int(row["year"]),
        "value": float(row["value"]),
        "unit": row["unit"],
        "statement": row["statement"],
        "section": row["section"],
        "source_file": row["source_file"],
    }

    if len(unique_values) > 1:
        result["warning"] = (
            f"Multiple different values found for this query: {list(unique_values)}. "
            f"Returning the first match — verify manually before trusting this."
        )

    return result


def calculate_metric(operation: str, company: str, metric: str, year_a: int, year_b: int = None) -> dict:
    data_a = get_financial_data(company, metric, year_a)
    if not data_a["found"]:
        return {"found": False, "message": f"Missing base data: {data_a['message']}"}

    if operation == "difference" or operation in ("growth_rate", "cagr"):
        if year_b is None:
            return {"found": False, "message": "This operation requires year_b."}

        data_b = get_financial_data(company, metric, year_b)
        if not data_b["found"]:
            return {"found": False, "message": f"Missing comparison data: {data_b['message']}"}

        value_a = data_a["value"]
        value_b = data_b["value"]

        if operation == "difference":
            result_value = value_b - value_a
        elif operation == "growth_rate":
            if value_a == 0:
                return {"found": False, "message": "Cannot compute growth rate — base year value is 0."}
            result_value = (value_b - value_a) / abs(value_a) * 100
        elif operation == "cagr":
            n_years = abs(year_b - year_a)
            if n_years == 0 or value_a <= 0:
                return {"found": False, "message": "CAGR requires distinct years and a positive base value."}
            result_value = ((value_b / value_a) ** (1 / n_years) - 1) * 100

        return {
            "found": True,
            "company": company,
            "metric": metric,
            "operation": operation,
            "year_a": year_a,
            "value_a": value_a,
            "year_b": year_b,
            "value_b": value_b,
            "result": round(result_value, 2),
            "unit": "percent" if operation in ("growth_rate", "cagr") else data_a["unit"],
        }

    return {"found": False, "message": f"Unknown operation: {operation}"}


def calculate_margin(company: str, numerator_metric: str, denominator_metric: str, year: int) -> dict:
    numerator = get_financial_data(company, numerator_metric, year)
    denominator = get_financial_data(company, denominator_metric, year)

    if not numerator["found"] or not denominator["found"]:
        return {
            "found": False,
            "message": f"Missing data — numerator found: {numerator['found']}, "
                       f"denominator found: {denominator['found']}"
        }

    if denominator["value"] == 0:
        return {"found": False, "message": "Cannot compute margin — denominator is 0."}

    margin = numerator["value"] / denominator["value"] * 100

    return {
        "found": True,
        "company": company,
        "numerator_metric": numerator_metric,
        "denominator_metric": denominator_metric,
        "year": year,
        "result": round(margin, 2),
        "unit": "percent",
    }