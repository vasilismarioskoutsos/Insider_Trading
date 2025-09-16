import pandas as pd
import numpy as np
import time

def _qtr_of_date(dt):
    m = dt.month
    return f"QTR{1 + (m-1)//3}"

def harvest_form4_range(get_daily_form4_csv, start_date, end_date, sleep_jitter=(0.05, 0.2)):
    """
    Runs your existing daily harvester across [start_date, end_date] (YYYY-MM-DD or datetime),
    concatenates results into one DataFrame (same schema as your daily output).
    """
    start = pd.to_datetime(start_date).date()
    end   = pd.to_datetime(end_date).date()

    out = []
    for dt in pd.date_range(start, end, freq="D"):
        ds = dt.strftime("%Y%m%d")
        q  = _qtr_of_date(dt)
        try:
            df = get_daily_form4_csv(ds, q, out_filename=None)  # your function already returns a DataFrame
            if df is not None and len(df):
                out.append(df)
        except Exception:
            pass
        # tiny jitter to avoid clockwork
        time.sleep(float(np.random.uniform(*sleep_jitter)))

    if not out:
        return pd.DataFrame()
    df_all = pd.concat(out, ignore_index=True)

    # (optional) drop exact duplicates if you rerun overlapping dates
    df_all = df_all.drop_duplicates(subset=[
        "issuerCik","ownerCik","securityTitle","transactionDate",
        "transactionCode","shares","pricePerShare","xmlUrl"
    ])
    return df_all

def owner_cross_issuer_history(df_all, owner_cik):
    """
    Filter the big table to a single reporting owner across ALL issuers.
    """
    d = df_all[df_all["ownerCik"].astype(str).str.lstrip("0").eq(str(int(owner_cik)))]
    return d.sort_values(["issuerCik","transactionDate","acceptedDatetime"], na_position="last")
