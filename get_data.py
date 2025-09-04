import io, csv, requests, pandas as pd
import re
import numpy as np
import requests_cache
from tenacity import retry, stop_after_attempt, wait_exponential_jitter
from lxml import etree
from ratelimiter import RateLimiter
from helpers import txt, classify_action, infer_unit_type, _format_yyyymmddhhmmss

'''This file gets all the SEC data for 1 day'''

UA = {"User-Agent": "Form4/1.0 (contact: doublethespeedchannel@gmail.com)"}  # SEC asks for this
BASE = "https://www.sec.gov"

_ACCEPTED_RE_TXT = re.compile(r"ACCEPTANCE-DATETIME:\s*([0-9]{14})")
_ACCEPTED_RE_HTML = re.compile(r"Accepted:\s*([0-9]{4}-[0-9]{2}-[0-9]{2}\s+[0-9]{2}:[0-9]{2}:[0-9]{2})")

DEBUG = True # set to true if you want debug prints
def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

# cached session, set longer TTLs for immutable resources
SESSION = requests_cache.CachedSession(
    cache_name = "sec_cache",
    backend = "sqlite",
    expire_after = 3600, # default 1 hour
    allowable_methods = ("GET",),
    allowable_codes = (200,),
    urls_expire_after = {
        re.compile(r".*/Archives/edgar/daily-index/.*/(?:company|form)\.\d+\.idx$"): 2 * 24 * 3600,  # 2 days
        re.compile(r".*/Archives/edgar/data/.*/index\.json$"): 365 * 24 * 3600, # 1 year
        re.compile(r".*\.xml$"): 365 * 24 * 3600, # 1 year
    },
)
SESSION.headers.update(UA)
SESSION.headers.update({"Accept-Encoding": "gzip, deflate"})

RL = RateLimiter(max_calls =2, period = 1.0)  # SEC request cap is 10 req/s max_calls has to be below that according to the guidelines

@retry(reraise=True, stop=stop_after_attempt(5), wait=wait_exponential_jitter(initial=0.5, max=10))
def _get(url: str, timeout: int = 30):
    """Cached, rate limited GET with retry on transient HTTP errors"""
    RL.wait()
    resp = SESSION.get(url, timeout=timeout)

    debug_print(f"GET {url} -> {resp.status_code}"
                f"{' (cache)' if getattr(resp, 'from_cache', False) else ''}")

    if getattr(resp, "from_cache", False):
        return resp
    if resp.status_code in {429, 502, 503, 504}:
        raise requests.exceptions.HTTPError(f"Transient HTTP {resp.status_code}", response=resp)
    if resp.status_code == 403:
        raise requests.exceptions.HTTPError(
            "403 Forbidden from SEC. Ensure a real contact in User-Agent and keep requests ≤10/sec.",
            response=resp,
        )
    return resp

# returns respond as a text, used for .idx files
def fetch_text(url):
    r = _get(url)
    r.raise_for_status()
    return r.text

# returns parsed JSON used for a folder’s index.json
def fetch_json(url):
    r = _get(url)
    r.raise_for_status()
    return r.json()

# returns raw bytes, used for XML so the parser can read it
def fetch_bytes(url):
    r = _get(url)
    r.raise_for_status()
    return r.content

def parse_daily_idx(idx_url):
    debug_print(f"\n[parse_daily_idx] fetching index: {idx_url}")
    text = fetch_text(idx_url)
    lines = [ln.rstrip("\n") for ln in text.splitlines()]

    # try to locate the header row ("Company Name  Form Type  CIK  Date Filed  File Name/Filename")
    try:
        header_i = next(i for i, ln in enumerate(lines) if re.search(r"^\s*Company\s+Name", ln, flags=re.I))
        dash_i = header_i + 1
        data_lines = [ln for ln in lines[dash_i + 1:] if ln.strip()]
        header = lines[header_i]
    except StopIteration:
        # Fallback: find dashed underline and assume everything after it is data
        dash_i = next(i for i, ln in enumerate(lines) if ln.strip().startswith("-----"))
        data_lines = [ln for ln in lines[dash_i + 1:] if ln.strip()]
        header = None

    if not data_lines:
        debug_print("[parse_daily_idx] No data lines found.")
        return pd.DataFrame(columns=["company", "form", "cik", "date_filed", "filename"])

    # fast path, some variants are already pipe-delimited
    if "|" in data_lines[0]:
        reader = csv.reader(io.StringIO("\n".join(data_lines)), delimiter="|")
        rows = [row for row in reader if len(row) >= 5]
        df = pd.DataFrame(rows, columns=["company", "form", "cik", "date_filed", "filename"])
        debug_print(f"[parse_daily_idx] Parsed {len(df)} rows (pipe-delimited). Head:\n{df.head(3)}")
        return df

    # fixed width split, try to infer column starts from the header
    def col_pos(pattern: str) -> int:
        if not header:
            return -1
        m = re.search(pattern, header, flags=re.I)
        return m.start() if m else -1

    starts = [
        col_pos(r"\bCompany\s+Name\b"),
        col_pos(r"\bForm\s+Type\b"),
        col_pos(r"\bCIK\b"),
        col_pos(r"\bDate\s+Filed\b"),
        col_pos(r"\bFile\s*Name\b|\bFilename\b"),
    ]

    # if any position is missing or not strictly increasing, use conservative defaults
    if any(s < 0 for s in starts) or any(b <= a for a, b in zip(starts, starts[1:])):
        starts = [0, 62, 74, 86, 98]
    ends = starts[1:] + [None]

    def split_fixed_width(ln: str):
        out = []
        for s, e in zip(starts, ends):
            out.append(ln[s:e].rstrip() if e is not None else ln[s:].rstrip())
        return out

    rows = [split_fixed_width(ln) for ln in data_lines if not ln.strip().startswith("-----")]
    df = pd.DataFrame(rows, columns=["company", "form", "cik", "date_filed", "filename"])

    # Skeep only plausible edgar paths
    df = df[df["filename"].str.contains(r"edgar/data/", case=False, na=False)]

    debug_print(
        f"[parse_daily_idx] Parsed {len(df)} rows (fixed-width). "
        f"Sample filenames: {df['filename'].head(3).tolist()}"
    )
    return df

def accepted_datetime_for_filing(accession_folder: str, filing_txt_url: str) -> str | None:
    '''
    Try to read ACCEPTANCE-DATETIME from the submission .txt
    If missing, fall back to the primary HTML page header ('Accepted: YYYY-MM-DD HH:MM:SS')
    '''
    try:
        txt = fetch_text(filing_txt_url)
        m = _ACCEPTED_RE_TXT.search(txt)
        if m:
            return _format_yyyymmddhhmmss(m.group(1))
    except Exception as e:
        debug_print(f"[accepted] txt fetch failed: {filing_txt_url} err={e}")

    # fallback: look for a primary .htm(l) in the folder and parse the header banner
    try:
        idx = fetch_json(accession_folder + "index.json")
        items = [it["name"] for it in idx["directory"]["item"]]
        # prefer names containing 'primary_doc' then any .htm/.html
        candidates = sorted(
            [n for n in items if n.lower().endswith((".htm", ".html"))],
            key=lambda n: (0 if "primary" in n.lower() else 1, n.lower())
        )
        if candidates:
            html = fetch_text(accession_folder + candidates[0])
            m = _ACCEPTED_RE_HTML.search(html)
            if m:
                return m.group(1)
    except Exception as e:
        debug_print(f"[accepted] html fallback failed for {accession_folder}: {e}")

    return None

# jump from the index row to the filing’s folder
def accession_folder_from_filename(filename: str):
    s = (filename or "").strip()
    # match with or without .txt at the end
    m = re.search(r"edgar/data/(\d{1,10})/([0-9\-]{18,})\.txt$", s, flags = re.IGNORECASE)
    if not m:
        m = re.search(r"edgar/data/(\d{1,10})/([0-9\-]{18,})", s, flags = re.IGNORECASE)
    if not m:
        raise ValueError(f"Unrecognized EDGAR filename: {filename!r}")
    cik = m.group(1)
    acc_nodash = m.group(2).replace("-", "")
    
    return f"{BASE}/Archives/edgar/data/{cik}/{acc_nodash}/"

# find the ownership XML inside a form 3/4/5 filing
def find_ownership_xml(folder_url):
    idx = fetch_json(folder_url + "index.json")
    items = [it["name"] for it in idx["directory"]["item"]]
    # common form 4 XML names
    candidates = []
    for name in items:
        low = name.lower()
        if low.endswith(".xml") and ("f345" in low or "primary_doc.xml" in low or "ownership.xml" in low or "form4.xml" in low):
            candidates.append(name)
    # if multiple, prefer primary_doc.xml or ownership.xml
    for prefer in ["xslF345X03/primary_doc.xml", "xslF345X05/primary_doc.xml", "ownership.xml", "form4.xml"]:
        for name in candidates:
            if name.endswith(prefer) or name == prefer or prefer.split("/")[-1] == name:
                return folder_url + name
    return folder_url + candidates[0] if candidates else None

# extract the trade rows from a form 4’s ownership XML.
def parse_form4_xml(xml_bytes):
    root = etree.fromstring(xml_bytes)  # XML
    ns = {} # not needed for these docs
    out = []
    for t in root.findall(".//nonDerivativeTransaction", namespaces=ns):
        out.append({
            "transactionDate": (t.findtext(".//transactionDate/value") or "").strip(),
            "transactionCode": (t.findtext(".//transactionCoding/transactionCode") or "").strip(),
            "shares": (t.findtext(".//transactionShares/value") or "").strip(),
            "price": (t.findtext(".//transactionPricePerShare/value") or "").strip(),
            "acqDisp": (t.findtext(".//transactionAcquiredDisposedCode/value") or "").strip(),
        })
    debug_print(f"[parse_form4_xml] Parsed {len(out)} non-derivative transactions")
    return out

# footnotes map ( for example price = 0 with explanation)
def collect_footnotes(root):
    foot = {}
    for n in root.findall(".//footnotes/footnote"):
        fid = n.get("id") or ""
        foot[fid] = "".join(n.itertext()).strip()
    return foot

# issuer/company being traded
def issuer_info(root):
    return {
        "issuerCik": txt(root, ".//issuer/issuerCik"),
        "issuerName": txt(root, ".//issuer/issuerName"),
        "issuerTicker": txt(root, ".//issuer/issuerTradingSymbol"),
    }

# reporting owners and roles
def reporting_owners(root):
    owners = []
    for ro in root.findall(".//reportingOwner"):
        owners.append({
            "ownerName": txt(ro, ".//reportingOwnerId/rptOwnerName"),
            "ownerCik": txt(ro, ".//reportingOwnerId/rptOwnerCik"),
            "isDirector": txt(ro, ".//reportingOwnerRelationship/isDirector"),
            "isOfficer": txt(ro, ".//reportingOwnerRelationship/isOfficer"),
            "officerTitle": txt(ro, ".//reportingOwnerRelationship/officerTitle"),
            "isTenPctOwner": txt(ro, ".//reportingOwnerRelationship/isTenPercentOwner"),
            "isOther": txt(ro, ".//reportingOwnerRelationship/isOther"),
            "otherText": txt(ro, ".//reportingOwnerRelationship/otherText"),
        })
    return owners

# collect any footnoteId references in a transaction
def footnote_ids(node):
    return ",".join(sorted(set([e.get("id") for e in node.findall(".//footnoteId") if e.get("id")])))

# unified row builder for non-derivative and derivative transactions
def transaction_rows(root, table_xpath, is_derivative: bool, issuer, owners, footnotes):
    rows = []
    for t in root.findall(table_xpath):
        txn_date = txt(t, ".//transactionDate/value")
        txn_code = txt(t, ".//transactionCoding/transactionCode")
        acq_disp = txt(t, ".//transactionAmounts/transactionAcquiredDisposedCode/value")
        shares = txt(t, ".//transactionAmounts/transactionShares/value")
        price = txt(t, ".//transactionAmounts/transactionPricePerShare/value")
        sec_title = txt(t, ".//securityTitle/value")
        strike = txt(t, ".//conversionOrExercisePrice/value") if is_derivative else ""
        fids = footnote_ids(t)
        ftxt = "; ".join(footnotes.get(fid, "") for fid in fids.split(",") if fid)

        for own in (owners or [{"ownerName": "", "ownerCik": ""}]):
            rows.append({
                "issuerCik": issuer["issuerCik"],
                "issuerName": issuer["issuerName"],
                "issuerTicker": issuer["issuerTicker"],
                "ownerName": own.get("ownerName", ""),
                "ownerCik": own.get("ownerCik", ""),
                "isDirector": own.get("isDirector", ""),
                "isOfficer": own.get("isOfficer", ""),
                "officerTitle": own.get("officerTitle", ""),
                "isTenPctOwner": own.get("isTenPctOwner", ""),
                "securityTitle": sec_title,
                "transactionDate": txn_date,
                "transactionCode": txn_code, # P (purchase), S (sale)
                "acqDisp": acq_disp, # A =Acquired, D = Disposed
                "shares": shares,
                "pricePerShare": price,
                "strikeOrExercisePrice": strike,
                "derivativeFlag": "Y" if is_derivative else "N",
                "footnoteIds": fids,
                "footnotesText": ftxt,
            })
    return rows

# extended parser that emits issuer/owner + both non-deriv and deriv transactions
def parse_form4_xml_extended(xml_bytes):
    '''
    If it is the stock itself = non-derivative
    If it grants/represents a right to get stock or can convert into stock = derivative
    '''
    root = etree.fromstring(xml_bytes)
    footnotes = collect_footnotes(root)
    issuer = issuer_info(root)
    owners = reporting_owners(root)

    rows = []
    rows += transaction_rows(root, ".//nonDerivativeTable/nonDerivativeTransaction", False, issuer, owners, footnotes)
    rows += transaction_rows(root, ".//derivativeTable/derivativeTransaction", True, issuer, owners, footnotes)
    return rows

def get_daily_form4_csv(date: str, quarter: str, out_filename: str | None = None):
    '''
    Writes a CSV with issuer, owner roles, transactions (non-deriv & deriv), prices, A/D flags, and footnotes
    For all daily transactions
    '''
    idx_url = f"{BASE}/Archives/edgar/daily-index/{date[:4]}/{quarter}/company.{date}.idx"
    debug_print(f"\n[get_daily_form4_csv] date={date} quarter={quarter}")
    debug_print(f"[get_daily_form4_csv] Index URL: {idx_url}")

    df = parse_daily_idx(idx_url)
    form4s = df[df["form"].str.strip().isin(["4", "4/A"])].reset_index(drop=True)
    debug_print(f"[get_daily_form4_csv] Found {len(form4s)} Form 4/4A filings")

    all_rows = []
    for _, row in form4s.iterrows():
        try:
            folder = accession_folder_from_filename(row["filename"])
        except Exception as e:
            debug_print(f"  [skip] bad filename={row.get('filename')} err={e}")
            continue
        xml_url = find_ownership_xml(folder)
        if not xml_url:
            debug_print(f"  [skip] No ownership XML in folder: {folder}")
            continue
        try:
            parsed_rows = parse_form4_xml_extended(fetch_bytes(xml_url))
            debug_print(f"  [ok] {len(parsed_rows)} rows from {xml_url}")
        except Exception as error:
            debug_print(f"  [error] Failed parsing {xml_url}: {error}")
            continue

        # attach filing level context from the index
        filing_txt_url = BASE + "/" + row["filename"].lstrip("/")
        form_type = (row["form"] or "").strip()

        # attach filing level context from the index
        for r in parsed_rows:
            r.update({
                "companyFromIndex": row["company"].strip(),
                "cikFromIndex": row["cik"].strip(),
                "filedDateFromIndex": row["date_filed"].strip(),
                "formTypeFromIndex": form_type,
                "accessionFolder": folder,
                "xmlUrl": xml_url,
                "filingTxtUrl": filing_txt_url,
            })
        all_rows.extend(parsed_rows)

    out = pd.DataFrame(all_rows)
    if out_filename is None:
        out_filename = f"form4_{date}.csv"

    # Fix dates that may appear malformed in the index (e.g., '2025082')
    out["filedDateFromIndex"] = pd.to_datetime(out["filedDateFromIndex"], errors="coerce").dt.date

    # Numerics
    for col in ["shares", "pricePerShare", "strikeOrExercisePrice"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Roles to proper booleans
    for b in ["isDirector", "isOfficer", "isTenPctOwner"]:
        out[b] = out[b].astype(str).str.upper().map({"TRUE": True, "FALSE": False})

    # Action, direction, open-market flags
    out["action"] = out.apply(lambda r: classify_action(r.get("transactionCode"), r.get("acqDisp")), axis=1)
    out["isOpenMarket"] = out["transactionCode"].isin(["P", "S"])
    out["direction"] = out["acqDisp"].map({"A": "ACQUIRED", "D": "DISPOSED"}).fillna("")

    # Cash value where price exists
    out["grossValue"] = (out["shares"] * out["pricePerShare"]).where(out["pricePerShare"] > 0)

    # isBuySell
    out["isBuySell"] = out["action"].isin(["BUY", "SELL"])

    # unitType inferred from securityTitle
    out["unitType"] = out["securityTitle"].map(infer_unit_type)

    # isAmendment from formTypeFromIndex
    out["isAmendment"] = (out["formTypeFromIndex"] == "4/A")

    # acceptedDatetime (cache per filing to avoid refetching)
    _accepted_cache = {}
    def _get_accept(r):
        key = r.get("filingTxtUrl") or r.get("accessionFolder")
        if not key:
            return None
        if key not in _accepted_cache:
            _accepted_cache[key] = accepted_datetime_for_filing(
                r.get("accessionFolder", ""), r.get("filingTxtUrl", "")
            )
        return _accepted_cache[key]

    out["acceptedDatetime"] = pd.to_datetime(out["acceptedDatetime"], errors="coerce")

    # netShares: sum signed shares within filing-day/security block
    # The user asked specifically for ownerCik + securityTitle + transactionDate.
    # We also keep it signed: A = +, D = -
    out["signedShares"] = np.where(out["acqDisp"].eq("A"), out["shares"],
                            np.where(out["acqDisp"].eq("D"), -out["shares"], 0))
    
    grp_keys = ["ownerCik", "issuerCik", "securityTitle", "transactionDate"]

    net = (
        out.groupby(grp_keys, dropna=False, as_index=False)["signedShares"]
          .sum()
          .rename(columns={"signedShares": "netShares"})
    )
    out = out.merge(net, on=grp_keys, how="left")
    
    debug_print(f"[get_daily_form4_csv] Total parsed rows: {len(out)}")
    out.to_csv(out_filename, index=False, encoding="utf-8-sig")
    print(f"Saved {len(out)} rows to {out_filename} from {len(form4s)} form 4 filings")

    return out