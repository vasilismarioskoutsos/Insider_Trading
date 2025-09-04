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
