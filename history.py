"""Form 4 history + pricing helpers

This module plugs into the existing form 4 harvester but can be imported and
used independently. It focuses on:

1) Building a cross-day history table (owner-agnostic),
2) Extracting a single owner's cross-issuer history from that table,
3) Scanning an issuer's /Archives folder to find filings that contain a given ownerCik, and
4) Attaching market prices and forward returns to Form 4 rows.

"""
from __future__ import annotations

import re
import time
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

def _qtr_of_date(dt) -> str:
    """Returns SEC quarter label"""
    dt = pd.to_datetime(dt)
    m = int(dt.month)
    return f"QTR{1 + (m - 1) // 3}"


def harvest_form4_range(get_daily_form4_csv: Callable[[str, str], pd.DataFrame], start_date, end_date, sleep_jitter: Sequence[float] = (0.05, 0.20)) -> pd.DataFrame:
    """
    Run the existing daily harvester across [start_date, end_date] (inclusive)
    and concatenate results (same schema as your daily output).

    Parameters
    ----------
    get_daily_form4_csv : callable(ds: YYYYMMDD, qtr: QTR#, out_filename=None) -> DataFrame
        Your daily function. It should accept out_filename=None and return a DataFrame.
    start_date, end_date : date-like
        Range bounds (inclusive). Strings OK.
    sleep_jitter : (low, high)
        Random sleep added per day to avoid clockwork patterns.

    Returns
    -------
    DataFrame
        Concatenated Form 4 rows; may be empty if no data.
    """
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()

    frames: list[pd.DataFrame] = []
    for dt in pd.date_range(start, end, freq="D"):
        ds = dt.strftime("%Y%m%d")
        quarter = _qtr_of_date(dt)
        try:
            df = get_daily_form4_csv(ds, quarter, out_filename=None)  # must return a DataFrame
            if df is not None and len(df):
                frames.append(df)
        except Exception:
            # continue
            pass
        # tiny jitter to avoid clockwork
        time.sleep(float(np.random.uniform(*sleep_jitter)))

    if not frames:
        return pd.DataFrame()

    df_all = pd.concat(frames, ignore_index=True)
    return dedupe_form4(df_all)


def dedupe_form4(df: pd.DataFrame) -> pd.DataFrame:
    """Drops exact duplicates likely to appear when ranges overlap."""
    subset = [
        "issuerCik",
        "ownerCik",
        "securityTitle",
        "transactionDate",
        "transactionCode",
        "shares",
        "pricePerShare",
        "xmlUrl",
    ]
    existing = [c for c in subset if c in df.columns]
    if not existing:
        return df
    return df.drop_duplicates(subset=existing)


def owner_cross_issuer_history(df_all: pd.DataFrame, owner_cik: str | int) -> pd.DataFrame:
    """Filter a table to all rows for a single reporting owner across issuers."""
    if df_all.empty:
        return df_all
    owner = str(int(owner_cik))
    mask = df_all["ownerCik"].astype(str).str.lstrip("0").eq(owner)
    sort_cols = [c for c in ["issuerCik", "transactionDate", "acceptedDatetime"] if c in df_all.columns]
    return df_all.loc[mask].sort_values(sort_cols, na_position="last")

def list_issuer_accessions(fetch_json: Callable[[str], dict], base_url: str, issuer_cik: str | int, max_items: int = 300) -> list[str]:
    """
    Return newest to oldest accession folder names for an issuer
    """
    json = fetch_json(f"{base_url}/Archives/edgar/data/{int(issuer_cik)}/index.json")
    items = json.get("directory", {}).get("item", [])
    accession = [it.get("name", "") for it in items if re.fullmatch(r"\d{18}", it.get("name", ""))]
    accession.sort(reverse=True)
    return accession[:max_items]

def find_submission_txt_url(fetch_json: Callable[[str], dict], folder_url: str) -> Optional[str]:
    """Find the master .txt file within a filing folder via its index.json"""
    idx = fetch_json(folder_url + "index.json")
    items = [it.get("name", "") for it in idx.get("directory", {}).get("item", [])]
    # 000XXXXXXXX-YY-ZZZZZZ.txt
    for item in items:
        if re.fullmatch(r"\d{10}-\d{2}-\d{6}\.txt", item):
            return folder_url + item
    for item in items:
        if item.lower().endswith(".txt"):
            return folder_url + item
    return None


def owner_form4s_in_issuer(
    fetch_json: Callable[[str], dict],
    fetch_bytes: Callable[[str], bytes],
    find_ownership_xml: Callable[[str], Optional[str]],
    parse_form4_xml_extended: Callable[[bytes], list[dict]],
    txt: Callable[[object, str], str],
    base_url: str,
    owner_cik: str | int,
    issuer_cik: str | int,
    *,
    accepted_datetime_for_filing: Optional[Callable[[str, str], Optional[str]]] = None,
    max_items: int = 300,
    sleep_jitter: Sequence[float] = (0.05, 0.20),
) -> pd.DataFrame:
    """
    Scan an issuer's recent accessions and return rows (same shape as your XML parser)
    for filings that mention the specified ownerCik.

    Network is delegated to your provided callables. Stays in /Archives.
    """
    from lxml import etree  # local import to avoid hard dep if unused

    out_rows: list[dict] = []
    base = f"{base_url}/Archives/edgar/data/{int(issuer_cik)}/"

    owner_norm = str(int(owner_cik))

    for acc in list_issuer_accessions(fetch_json, base_url, issuer_cik, max_items=max_items):
        folder = f"{base}{acc}/"
        xml_url = find_ownership_xml(folder)
        if not xml_url:
            continue
        try:
            time.sleep(float(np.random.uniform(*sleep_jitter)))
            xml_bytes = fetch_bytes(xml_url)
            root = etree.fromstring(xml_bytes)
            owner_ciks = [txt(ro, ".//reportingOwnerId/rptOwnerCik") for ro in root.findall(".//reportingOwner")]
            if owner_norm not in {c.lstrip("0") for c in owner_ciks if c}:
                continue

            rows = parse_form4_xml_extended(xml_bytes)
            filing_txt_url = find_submission_txt_url(fetch_json, folder)
            for r in rows:
                r.update(
                    {
                        "issuerCik": str(int(issuer_cik)),
                        "issuerName": txt(root, ".//issuer/issuerName"),
                        "issuerTicker": txt(root, ".//issuer/issuerTradingSymbol"),
                        "accessionFolder": folder,
                        "xmlUrl": xml_url,
                        "filingTxtUrl": filing_txt_url or "",
                    }
                )
                if accepted_datetime_for_filing and filing_txt_url:
                    r["acceptedDatetime"] = accepted_datetime_for_filing(folder, filing_txt_url)
            out_rows.extend(rows)
        except Exception:
            # Skip problematic filings; keep going
            continue

    df = pd.DataFrame(out_rows)
    if not df.empty and "acceptedDatetime" in df.columns:
        df["acceptedDatetime"] = pd.to_datetime(df["acceptedDatetime"], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Pricing helpers (attach market prices / forward returns)
# ---------------------------------------------------------------------------

def price_on_date(px: pd.DataFrame, datetime, *, price_col: str = "Adj Close", policy: str = "next"):
    """
    Look up price on/around date d from a trading-day-indexed DataFrame.

    policy:
      - "next": next trading day if d isn't in index
      - "prev": previous trading day if d isn't in index
      - "exact": NaN if d isn't in index
    """
    datetime = pd.to_datetime(datetime).normalize()
    if price_col not in px.columns:
        raise ValueError(f"{price_col} not in px")
    idx = pd.to_datetime(px.index).normalize()

    if policy == "exact":
        try:
            return float(px.loc[datetime, price_col])
        except Exception:
            return float("nan")

    i = idx.searchsorted(datetime, side="left")
    if policy == "next":
        if i >= len(idx):
            return float("nan")
        return float(px.iloc[i][price_col])
    elif policy == "prev":
        if i == len(idx) or idx[i] != datetime:
            i -= 1
        if i < 0:
            return float("nan")
        return float(px.iloc[i][price_col])
    else:
        raise ValueError("policy must be one of: next, prev, exact")


def _effective_trade_date(row: pd.Series, cutoff_hour: int = 16):
    """
    Same-day vs next business day logic based on filing acceptance time.
    If acceptedDatetime is >= cutoff_hour (local to timestamp), start next BDay.
    Falls back to transactionDate if acceptedDatetime missing.
    """
    tx = pd.to_datetime(row.get("transactionDate")).normalize()
    acc = pd.to_datetime(row.get("acceptedDatetime"))
    if pd.isna(acc):
        return tx
    return tx + BDay(1) if getattr(acc, "hour", 0) >= int(cutoff_hour) else tx


def attach_trade_day_close(df_form4: pd.DataFrame,  px_loader: Callable[[str], pd.DataFrame], *, price_col: str = "Adj Close", policy: str = "next",) -> pd.DataFrame:
    """
    For each row (issuerTicker, transactionDate), add market close on that day.
    Expects px_loader(ticker) -> DataFrame indexed by trading days with `price_col`.
    """
    if df_form4.empty:
        return df_form4.assign(close_on_trade=np.nan)

    out = df_form4.copy()
    out["close_on_trade"] = np.nan

    # Group by ticker to avoid repeated loads
    tickers = out["issuerTicker"].astype(str).str.upper().fillna("") if "issuerTicker" in out.columns else pd.Series(dtype=str)
    for tkr, g in out.groupby(tickers, dropna=False):
        if not tkr or tkr == "NAN":
            continue
        try:
            px = px_loader(tkr)
            px = px.sort_index()
            px.index = pd.to_datetime(px.index).normalize()
        except Exception:
            continue

        for i, r in g.iterrows():
            d = r.get("transactionDate")
            out.at[i, "close_on_trade"] = price_on_date(px, d, price_col=price_col, policy=policy)

    return out


def attach_prices_and_returns(
    df: pd.DataFrame,
    px_loader: Callable[[str], pd.DataFrame],
    *,
    price_col: str = "Adj Close",
    horizons: Sequence[int] = (1, 5, 21, 63),
    use_effective_date: bool = True,
    cutoff_hour: int = 16,
) -> pd.DataFrame:
    """
    Attach `mkt_t0` (adjusted close on effective day) and forward returns r{h}d.

    Parameters
    ----------
    df : DataFrame
        Must include issuerTicker, transactionDate, acceptedDatetime (optional).
    px_loader : function(ticker)-> DataFrame with index of trading dates and a column `price_col`.
    use_effective_date : bool
        If True, use next BDay when acceptedDatetime >= cutoff_hour; else same-day.
    cutoff_hour : int
        Hour threshold for rolling to next BDay.
    """
    if df.empty:
        cols = {"mkt_t0": np.nan}
        cols.update({f"r{h}d": np.nan for h in horizons})
        return df.assign(**cols)

    out = df.copy()

    # Effective starting date per row
    if use_effective_date:
        out["effDate"] = out.apply(lambda r: _effective_trade_date(r, cutoff_hour=cutoff_hour), axis=1)
    else:
        out["effDate"] = pd.to_datetime(out["transactionDate"]).dt.normalize()

    out["mkt_t0"] = np.nan
    for h in horizons:
        out[f"r{h}d"] = np.nan

    # Iterate per ticker to minimize API calls
    tickers = out["issuerTicker"].astype(str).str.upper().fillna("") if "issuerTicker" in out.columns else pd.Series(dtype=str)
    for ticker, g in out.groupby(tickers, dropna=False):
        if not ticker or ticker == "NAN":
            continue
        try:
            px = px_loader(ticker)
            px = px.sort_index()
            px.index = pd.to_datetime(px.index).normalize()
            if price_col not in px.columns:
                raise ValueError(f"{price_col} missing for {ticker}")
        except Exception:
            continue

        # Helper to get price at offset from eff date
        def price_at_offset(d, n):
            d = pd.to_datetime(d).normalize()
            i = px.index.searchsorted(d, side="left") + int(n)
            if 0 <= i < len(px):
                return float(px.iloc[i][price_col])
            return float("nan")

        for i, r in g.iterrows():
            d0 = r["effDate"]
            base = price_at_offset(d0, 0)
            out.at[i, "mkt_t0"] = base
            if np.isfinite(base) and base > 0:
                for h in horizons:
                    nxt = price_at_offset(d0, h)
                    out.at[i, f"r{h}d"] = (nxt / base - 1.0) if np.isfinite(nxt) else np.nan

    # Optional sanity: execution vs market bps (will be NaN for $0 awards)
    if "pricePerShare" in out.columns:
        out["exec_vs_mkt_bps"] = (out["pricePerShare"] / out["mkt_t0"] - 1.0) * 1e4

    return out

def build_owner_history_with_prices(
    get_daily_form4_csv: Callable[[str, str], pd.DataFrame],
    owner_cik: str | int,
    start_date,
    end_date,
    px_loader: Callable[[str], pd.DataFrame],
    *,
    price_col: str = "Adj Close",
    horizons: Sequence[int] = (1, 5, 21, 63),
    sleep_jitter: Sequence[float] = (0.05, 0.20),
    use_effective_date: bool = True,
    cutoff_hour: int = 16,
) -> pd.DataFrame:
    """
    One-liner workflow: harvest range → filter owner → attach prices/returns.
    """
    df_all = harvest_form4_range(get_daily_form4_csv, start_date, end_date, sleep_jitter=sleep_jitter)
    df_owner = owner_cross_issuer_history(df_all, owner_cik)
    return attach_prices_and_returns(
        df_owner,
        px_loader,
        price_col=price_col,
        horizons=horizons,
        use_effective_date=use_effective_date,
        cutoff_hour=cutoff_hour,
    )