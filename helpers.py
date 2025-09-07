import re

# safe text helper
def txt(node, path):
    v = node.findtext(path)
    return v.strip() if v else ""

def classify_action(code: str, acq_disp: str) -> str:
    code = (code or "").upper().strip()
    acq_disp = (acq_disp or "").upper().strip()
    if code == "P": return "BUY"
    if code == "S": return "SELL"
    if code in {"M", "C"}: return "EXERCISE/CONVERSION"
    if code == "A": return "GRANT/AWARD"
    if code == "G": return "GIFT"
    if code == "F": return "TAX-WITHHOLD/NET-SHARE"
    return "OTHER"

def infer_unit_type(security_title: str) -> str:
    t = (security_title or "").lower()
    if "restricted stock unit" in t or re.search(r"\brsu\b", t): return "RSU"
    if "performance stock unit" in t or re.search(r"\bpsu\b", t): return "PSU"
    if "stock option" in t or "right to buy" in t or re.search(r"\boption\b", t): return "Option"
    if "warrant" in t: return "Warrant"
    if "deferred stock" in t: return "Deferred Stock"
    if "restricted stock" in t: return "Restricted Stock"
    if "preferred" in t: return "Preferred"
    if "common stock" in t: return "Common Stock"
    if "class" in t and "common" in t: return "Common Stock"
    return "Other"

def _format_yyyymmddhhmmss(s: str) -> str:
    # 20250820123456 -> 2025-08-20 12:34:56
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]} {s[8:10]}:{s[10:12]}:{s[12:14]}"

# jump from the index row to the filing’s folder
def accession_folder_from_filename(filename: str, BASE: str):
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