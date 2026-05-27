"""
csv_analyser.py
================
Usage:
    python csv_analyser.py <path_to_csv_file> [output_txt_file]

Example:
    python csv_analyser.py data.csv data_analysis.txt

If no output file is given, it saves as  <csv_name>_analysis.txt

then use this analyser file to generate

Data cleaning code — imputation, outlier treatment, type fixes
EDA code — all the plots (histograms, boxplots, heatmaps, pairplots)
Feature engineering suggestions
Model selection + accuracy scoring based on what the target column looks like
"""

import sys
import os
import math
import csv
from collections import Counter
from datetime import datetime

# ── try to import pandas; fall back to stdlib csv if unavailable ──────────────
try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def divider(char="=", width=70):
    return char * width

def section(title):
    return f"\n{divider()}\n  {title}\n{divider()}\n"

def percentile(sorted_data, p):
    """Pure-python percentile (linear interpolation)."""
    n = len(sorted_data)
    if n == 0:
        return None
    idx = (p / 100) * (n - 1)
    lo = int(idx)
    hi = lo + 1
    if hi >= n:
        return sorted_data[-1]
    frac = idx - lo
    return sorted_data[lo] + frac * (sorted_data[hi] - sorted_data[lo])

def basic_stats(values):
    """Return dict of stats for a list of floats (NaN already removed)."""
    n = len(values)
    if n == 0:
        return {}
    s = sorted(values)
    mean = sum(s) / n
    variance = sum((x - mean) ** 2 for x in s) / n
    std = math.sqrt(variance)
    q1  = percentile(s, 25)
    med = percentile(s, 50)
    q3  = percentile(s, 75)
    iqr = q3 - q1
    return {
        "count":  n,
        "mean":   mean,
        "std":    std,
        "min":    s[0],
        "25%":    q1,
        "50%":    med,
        "75%":    q3,
        "max":    s[-1],
        "iqr":    iqr,
        "skew":   skewness(s, mean, std),
    }

def skewness(sorted_vals, mean, std):
    """Pearson's moment skewness."""
    n = len(sorted_vals)
    if n < 3 or std == 0:
        return 0.0
    return (sum((x - mean) ** 3 for x in sorted_vals) / n) / (std ** 3)

def detect_outliers_iqr(values, q1, q3, iqr):
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    out = [x for x in values if x < lo or x > hi]
    return out, lo, hi

def try_numeric(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Analysis using pandas (preferred)
# ─────────────────────────────────────────────────────────────────────────────

def analyse_with_pandas(csv_path):
    lines = []

    df = pd.read_csv(csv_path, low_memory=False)
    n_rows, n_cols = df.shape

    # ── 1. Overview ──────────────────────────────────────────────────────────
    lines.append(section("1. DATASET OVERVIEW"))
    lines.append(f"  File            : {os.path.basename(csv_path)}")
    lines.append(f"  Rows            : {n_rows:,}")
    lines.append(f"  Columns         : {n_cols}")
    lines.append(f"  Total cells     : {n_rows * n_cols:,}")
    lines.append(f"  Duplicate rows  : {df.duplicated().sum():,}")
    lines.append(f"  Memory usage    : {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    lines.append(f"\n  Columns : {list(df.columns)}")

    # ── 2. Data types ────────────────────────────────────────────────────────
    lines.append(section("2. DATA TYPES & COLUMN INFO"))
    lines.append(f"  {'Column':<30} {'Dtype':<15} {'Non-Null':>10} {'Null':>8} {'Null%':>8}")
    lines.append("  " + "-" * 75)
    for col in df.columns:
        null_cnt  = df[col].isna().sum()
        non_null  = n_rows - null_cnt
        null_pct  = 100 * null_cnt / n_rows if n_rows else 0
        lines.append(f"  {col:<30} {str(df[col].dtype):<15} {non_null:>10,} {null_cnt:>8,} {null_pct:>7.2f}%")

    # ── 3. Missing value analysis ────────────────────────────────────────────
    lines.append(section("3. MISSING VALUE ANALYSIS"))
    null_df = df.isnull().sum()
    null_df = null_df[null_df > 0].sort_values(ascending=False)
    if null_df.empty:
        lines.append("  ✅  No missing values found.")
    else:
        lines.append(f"  Columns with missing values : {len(null_df)}")
        lines.append(f"  Total missing cells         : {null_df.sum():,}")
        lines.append(f"  Overall missing %           : {100*null_df.sum()/(n_rows*n_cols):.2f}%\n")
        lines.append(f"  {'Column':<30} {'Missing':>10} {'Missing%':>10}  Severity")
        lines.append("  " + "-" * 65)
        for col, cnt in null_df.items():
            pct = 100 * cnt / n_rows
            sev = "🔴 HIGH" if pct > 50 else "🟡 MEDIUM" if pct > 20 else "🟢 LOW"
            lines.append(f"  {col:<30} {cnt:>10,} {pct:>9.2f}%  {sev}")

    # ── 4. Numeric column statistics ─────────────────────────────────────────
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    lines.append(section("4. NUMERIC COLUMNS — DESCRIPTIVE STATISTICS"))
    if not num_cols:
        lines.append("  No numeric columns found.")
    else:
        lines.append(f"  Numeric columns ({len(num_cols)}): {num_cols}\n")
        for col in num_cols:
            s = df[col].dropna()
            lines.append(f"  ── {col} ──")
            if len(s) == 0:
                lines.append("     All values are null.\n")
                continue
            desc = s.describe(percentiles=[.25, .5, .75])
            lines.append(f"     count   : {int(desc['count']):,}")
            lines.append(f"     mean    : {desc['mean']:.4f}")
            lines.append(f"     std     : {desc['std']:.4f}")
            lines.append(f"     min     : {desc['min']:.4f}")
            lines.append(f"     25%     : {desc['25%']:.4f}")
            lines.append(f"     50%     : {desc['50%']:.4f}")
            lines.append(f"     75%     : {desc['75%']:.4f}")
            lines.append(f"     max     : {desc['max']:.4f}")
            lines.append(f"     skew    : {float(s.skew()):.4f}  "
                        f"({'right-skewed' if s.skew()>0.5 else 'left-skewed' if s.skew()<-0.5 else 'approx. normal'})")
            lines.append(f"     kurtosis: {float(s.kurtosis()):.4f}")
            lines.append(f"     zeros   : {(s == 0).sum():,}")
            lines.append(f"     negatives: {(s < 0).sum():,}")
            lines.append(f"     unique  : {s.nunique():,}\n")

    # ── 5. Outlier detection ─────────────────────────────────────────────────
    lines.append(section("5. OUTLIER DETECTION (IQR method, 1.5×IQR rule)"))
    if not num_cols:
        lines.append("  No numeric columns to check.")
    else:
        lines.append(f"  {'Column':<30} {'Outliers':>10} {'Outlier%':>10}  {'Lower fence':>14}  {'Upper fence':>14}")
        lines.append("  " + "-" * 85)
        for col in num_cols:
            s = df[col].dropna()
            if len(s) == 0:
                continue
            q1  = s.quantile(0.25)
            q3  = s.quantile(0.75)
            iqr = q3 - q1
            lo  = q1 - 1.5 * iqr
            hi  = q3 + 1.5 * iqr
            out = ((s < lo) | (s > hi)).sum()
            pct = 100 * out / len(s)
            lines.append(f"  {col:<30} {out:>10,} {pct:>9.2f}%  {lo:>14.4f}  {hi:>14.4f}")

    # ── 6. Categorical column analysis ───────────────────────────────────────
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    lines.append(section("6. CATEGORICAL COLUMNS — VALUE DISTRIBUTION"))
    if not cat_cols:
        lines.append("  No categorical columns found.")
    else:
        lines.append(f"  Categorical columns ({len(cat_cols)}): {cat_cols}\n")
        for col in cat_cols:
            s   = df[col]
            vc  = s.value_counts(dropna=False)
            lines.append(f"  ── {col} ──")
            lines.append(f"     unique values  : {s.nunique():,}")
            lines.append(f"     null count     : {s.isna().sum():,}")
            lines.append(f"     top 10 values  :")
            for val, cnt in vc.head(10).items():
                pct = 100 * cnt / n_rows
                lines.append(f"       {str(val):<35} {cnt:>8,}  ({pct:.2f}%)")
            if len(vc) > 10:
                lines.append(f"       ... and {len(vc)-10} more unique values")
            lines.append("")

    # ── 7. Possible datetime columns ─────────────────────────────────────────
    lines.append(section("7. POTENTIAL DATETIME COLUMNS"))
    dt_candidates = []
    for col in cat_cols:
        sample = df[col].dropna().head(100)
        parsed = 0
        for v in sample:
            try:
                pd.to_datetime(v)
                parsed += 1
            except Exception:
                pass
        if parsed / max(len(sample), 1) > 0.7:
            dt_candidates.append(col)
    if dt_candidates:
        for col in dt_candidates:
            lines.append(f"  📅 '{col}' looks like a datetime column (>70% values parseable).")
    else:
        lines.append("  No obvious datetime columns detected (check manually).")

    # ── 8. Correlation analysis ───────────────────────────────────────────────
    lines.append(section("8. CORRELATION MATRIX (numeric columns)"))
    if len(num_cols) < 2:
        lines.append("  Need at least 2 numeric columns for correlation.")
    else:
        corr = df[num_cols].corr()
        # Print as a readable table
        col_w = 14
        header = " " * 30 + "".join(f"{c[:col_w]:>{col_w}}" for c in num_cols)
        lines.append("  " + header)
        lines.append("  " + "-" * (30 + col_w * len(num_cols)))
        for row_col in num_cols:
            row_str = f"  {row_col:<30}"
            for c in num_cols:
                val = corr.loc[row_col, c]
                row_str += f"{val:>{col_w}.3f}"
            lines.append(row_str)

        # Highlight high correlations
        lines.append("\n  📌 High correlations (|r| > 0.7, excluding self):")
        found_high = False
        for i, c1 in enumerate(num_cols):
            for c2 in num_cols[i+1:]:
                r = corr.loc[c1, c2]
                if abs(r) > 0.7:
                    lines.append(f"     {c1}  ↔  {c2}  :  r = {r:.4f}")
                    found_high = True
        if not found_high:
            lines.append("     None found.")

    # ── 9. Data quality summary ───────────────────────────────────────────────
    lines.append(section("9. DATA QUALITY SUMMARY & RECOMMENDATIONS"))
    issues = []

    if df.duplicated().sum() > 0:
        issues.append(f"  ⚠️  {df.duplicated().sum()} duplicate rows → consider df.drop_duplicates()")

    for col in null_df.index if not null_df.empty else []:
        pct = 100 * null_df[col] / n_rows
        if pct > 50:
            issues.append(f"  ⚠️  '{col}' has {pct:.1f}% missing → consider dropping this column")
        elif pct > 20:
            issues.append(f"  ⚠️  '{col}' has {pct:.1f}% missing → impute with median/mode or model-based imputation")
        else:
            issues.append(f"  ℹ️  '{col}' has {pct:.1f}% missing → safe to impute with mean/median/mode")

    for col in num_cols:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        out_cnt = ((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum()
        if out_cnt / len(s) > 0.05:
            issues.append(f"  ⚠️  '{col}' has {out_cnt} outliers ({100*out_cnt/len(s):.1f}%) → cap/winsorise or log-transform")
        if abs(float(s.skew())) > 1.0:
            issues.append(f"  ℹ️  '{col}' is highly skewed (skew={s.skew():.2f}) → consider log/sqrt transform")

    for col in cat_cols:
        if col in dt_candidates:
            issues.append(f"  ℹ️  '{col}' → parse as datetime with pd.to_datetime()")
        unique_ratio = df[col].nunique() / n_rows
        if unique_ratio > 0.9:
            issues.append(f"  ℹ️  '{col}' has very high cardinality ({df[col].nunique()} unique) → might be an ID column")

    if issues:
        for issue in issues:
            lines.append(issue)
    else:
        lines.append("  ✅  No major data quality issues detected.")

    lines.append(f"\n{divider()}")
    lines.append(f"  Analysis generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Tool               : csv_analyser.py  (pandas mode)")
    lines.append(divider())

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Analysis using stdlib only (fallback when pandas is unavailable)
# ─────────────────────────────────────────────────────────────────────────────

def analyse_without_pandas(csv_path):
    lines = []

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
        col_names = reader.fieldnames or []

    n_rows = len(all_rows)
    n_cols = len(col_names)

    # ── 1. Overview ──────────────────────────────────────────────────────────
    lines.append(section("1. DATASET OVERVIEW"))
    lines.append(f"  File       : {os.path.basename(csv_path)}")
    lines.append(f"  Rows       : {n_rows:,}")
    lines.append(f"  Columns    : {n_cols}")
    lines.append(f"  Total cells: {n_rows * n_cols:,}")
    lines.append(f"\n  Columns : {col_names}")

    # ── Build per-column buckets ──────────────────────────────────────────────
    col_raw   = {c: [r[c] for r in all_rows] for c in col_names}
    col_nums  = {}   # col → list of floats (nulls removed)
    col_nulls = {}   # col → null count

    for c in col_names:
        nulls = sum(1 for v in col_raw[c] if v is None or v.strip() == "")
        col_nulls[c] = nulls
        nums = [try_numeric(v) for v in col_raw[c] if v and v.strip() != ""]
        nums = [x for x in nums if x is not None]
        col_nums[c] = nums

    # ── 2. Data types ────────────────────────────────────────────────────────
    lines.append(section("2. COLUMN INFO & NULL COUNTS"))
    lines.append(f"  {'Column':<30} {'Type':<12} {'Non-Null':>10} {'Null':>8} {'Null%':>8}")
    lines.append("  " + "-" * 72)
    for c in col_names:
        null_c = col_nulls[c]
        non_n  = n_rows - null_c
        pct    = 100 * null_c / n_rows if n_rows else 0
        dtype  = "numeric" if col_nums[c] and len(col_nums[c]) > n_rows * 0.5 else "text"
        lines.append(f"  {c:<30} {dtype:<12} {non_n:>10,} {null_c:>8,} {pct:>7.2f}%")

    # ── 3. Missing values ────────────────────────────────────────────────────
    lines.append(section("3. MISSING VALUE ANALYSIS"))
    missing = {c: v for c, v in col_nulls.items() if v > 0}
    if not missing:
        lines.append("  ✅  No missing values found.")
    else:
        lines.append(f"  Columns with missing values : {len(missing)}")
        for c, cnt in sorted(missing.items(), key=lambda x: -x[1]):
            pct = 100 * cnt / n_rows
            sev = "🔴 HIGH" if pct > 50 else "🟡 MEDIUM" if pct > 20 else "🟢 LOW"
            lines.append(f"  {c:<30} {cnt:>8,}  ({pct:.2f}%)  {sev}")

    # ── 4. Numeric stats ─────────────────────────────────────────────────────
    lines.append(section("4. NUMERIC COLUMNS — DESCRIPTIVE STATISTICS"))
    numeric_cols = [c for c in col_names if col_nums[c] and len(col_nums[c]) > n_rows * 0.3]
    if not numeric_cols:
        lines.append("  No numeric columns detected.")
    else:
        for c in numeric_cols:
            st = basic_stats(col_nums[c])
            if not st:
                continue
            lines.append(f"  ── {c} ──")
            for k, v in st.items():
                lines.append(f"     {k:<10}: {v:.4f}")
            lines.append("")

    # ── 5. Outliers ──────────────────────────────────────────────────────────
    lines.append(section("5. OUTLIER DETECTION (IQR method)"))
    if not numeric_cols:
        lines.append("  No numeric columns to check.")
    else:
        lines.append(f"  {'Column':<30} {'Outliers':>10} {'Outlier%':>10}  {'Lower':>12}  {'Upper':>12}")
        lines.append("  " + "-" * 80)
        for c in numeric_cols:
            st = basic_stats(col_nums[c])
            if not st or st["iqr"] == 0:
                continue
            out, lo, hi = detect_outliers_iqr(col_nums[c], st["25%"], st["75%"], st["iqr"])
            pct = 100 * len(out) / st["count"]
            lines.append(f"  {c:<30} {len(out):>10,} {pct:>9.2f}%  {lo:>12.4f}  {hi:>12.4f}")

    # ── 6. Categorical ───────────────────────────────────────────────────────
    cat_cols = [c for c in col_names if c not in numeric_cols]
    lines.append(section("6. CATEGORICAL COLUMNS — VALUE DISTRIBUTION"))
    for c in cat_cols:
        vals = [v for v in col_raw[c] if v and v.strip()]
        cnt  = Counter(vals)
        lines.append(f"  ── {c} ──")
        lines.append(f"     unique values : {len(cnt)}")
        lines.append(f"     null count    : {col_nulls[c]}")
        lines.append(f"     top 10 values :")
        for val, freq in cnt.most_common(10):
            pct = 100 * freq / n_rows
            lines.append(f"       {str(val):<35} {freq:>8,}  ({pct:.2f}%)")
        lines.append("")

    lines.append(f"\n{divider()}")
    lines.append(f"  Analysis generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Tool               : csv_analyser.py  (stdlib mode — install pandas for full analysis)")
    lines.append(divider())

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    csv_path = sys.argv[1]
    if not os.path.exists(csv_path):
        print(f"ERROR: File not found → {csv_path}")
        sys.exit(1)

    base    = os.path.splitext(os.path.basename(csv_path))[0]
    out_txt = sys.argv[2] if len(sys.argv) > 2 else f"{base}_analysis.txt"

    print(f"\n📊  Analysing: {csv_path}")
    print(f"   Mode: {'pandas (full analysis)' if HAS_PANDAS else 'stdlib (basic analysis — pip install pandas for full)'}")

    if HAS_PANDAS:
        report = analyse_with_pandas(csv_path)
    else:
        report = analyse_without_pandas(csv_path)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n✅  Report saved → {out_txt}")
    print(f"   Lines written: {report.count(chr(10))}")
    print(f"\n   👉 Now send '{out_txt}' to Claude instead of the raw CSV.\n")


if __name__ == "__main__":
    main()
