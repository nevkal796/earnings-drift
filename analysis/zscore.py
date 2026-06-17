import pandas as pd
from scipy import stats
from ingestion.loader import get_connection


def load_dataset() -> pd.DataFrame:
    """Load full dataset same as correlations.py."""
    conn = get_connection()
    query = """
        SELECT
            c.ticker,
            c.sector,
            f.filed_at,
            lf.hedging_score,
            lf.uncertainty_score,
            lf.specificity_score,
            lf.fog_index,
            lf.finbert_positive,
            lf.finbert_negative,
            ps.return_30d,
            ps.return_60d,
            ps.return_90d,
            t.word_count
        FROM linguistic_features lf
        JOIN transcripts t ON t.id = lf.transcript_id
        JOIN filings f ON f.id = t.filing_id
        JOIN companies c ON c.id = f.company_id
        JOIN price_snapshots ps
            ON ps.company_id = c.id
            AND ps.filing_date = f.filed_at
        WHERE ps.return_30d IS NOT NULL
        AND lf.hedging_score < 30
        AND lf.fog_index < 35
        AND t.word_count > 200
        ORDER BY c.ticker, f.filed_at
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df


FEATURES = [
    "hedging_score",
    "uncertainty_score",
    "specificity_score",
    "fog_index",
    "finbert_positive",
    "finbert_negative",
]


def add_zscores(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each feature, compute z-score relative to that
    company's own historical mean and std.
    This normalizes for baseline communication style differences
    between companies — a hedging score of 8 means different things
    for AAPL vs JPM.
    """
    df = df.copy()
    for feature in FEATURES:
        z_col = f"{feature}_z"
        df[z_col] = df.groupby("ticker")[feature].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )
    return df


def run_zscore_correlations(df: pd.DataFrame):
    """Run correlations on z-scored features vs returns."""
    targets = ["return_30d", "return_60d", "return_90d"]
    z_features = [f"{f}_z" for f in FEATURES]

    results = []
    for feature in z_features:
        for target in targets:
            clean = df[[feature, target]].dropna()
            if len(clean) < 5:
                continue
            corr, pvalue = stats.spearmanr(clean[feature], clean[target])
            results.append({
                "feature": feature,
                "target": target,
                "n": len(clean),
                "correlation": round(corr, 4),
                "pvalue": round(pvalue, 4),
                "significant": pvalue < 0.05,
            })

    return pd.DataFrame(results)


def run_sector_correlations(df: pd.DataFrame):
    """Run z-scored correlations broken down by sector."""
    results = []
    for sector in df["sector"].unique():
        sector_df = df[df["sector"] == sector]
        if len(sector_df) < 10:
            continue
        for feature in [f"{f}_z" for f in FEATURES]:
            for target in ["return_30d", "return_90d"]:
                clean = sector_df[[feature, target]].dropna()
                if len(clean) < 5:
                    continue
                corr, pvalue = stats.spearmanr(clean[feature], clean[target])
                results.append({
                    "sector": sector,
                    "feature": feature,
                    "target": target,
                    "n": len(clean),
                    "correlation": round(corr, 4),
                    "pvalue": round(pvalue, 4),
                    "significant": pvalue < 0.05,
                })
    return pd.DataFrame(results)


def print_table(results_df: pd.DataFrame, title: str, group_col: str = None):
    print(f"\n{'='*65}")
    print(f" {title}")
    print(f"{'='*65}")
    if group_col:
        for group in results_df[group_col].unique():
            print(f"\n  [{group}]")
            subset = results_df[results_df[group_col] == group]
            for _, row in subset.iterrows():
                sig = "✓" if row["significant"] else ""
                print(f"    {row['feature']:<28} {row['target']:<12} "
                      f"n={row['n']:<4} r={row['correlation']:>7.4f} "
                      f"p={row['pvalue']:.4f} {sig}")
    else:
        for _, row in results_df.iterrows():
            sig = "✓" if row["significant"] else ""
            print(f"  {row['feature']:<28} {row['target']:<12} "
                  f"n={row['n']:<4} r={row['correlation']:>7.4f} "
                  f"p={row['pvalue']:.4f} {sig}")


if __name__ == "__main__":
    print("Loading dataset...")
    df = load_dataset()
    print(f"Dataset: {len(df)} observations, "
          f"{df['ticker'].nunique()} companies, "
          f"{df['sector'].nunique()} sectors")

    print("\nAdding z-scores...")
    df = add_zscores(df)

    # Overall z-scored correlations
    overall = run_zscore_correlations(df)
    print_table(overall, "Z-SCORED CORRELATIONS — All Companies")

    # Significant findings
    sig = overall[overall["significant"]]
    if len(sig):
        print(f"\n{'='*65}")
        print(" SIGNIFICANT Z-SCORED FINDINGS (p < 0.05)")
        print(f"{'='*65}")
        for _, row in sig.iterrows():
            direction = "↑ positive" if row["correlation"] > 0 else "↓ negative"
            print(f"  {row['feature']} → {row['target']}: "
                  f"r={row['correlation']} p={row['pvalue']} {direction}")
    else:
        print("\n  No significant z-scored correlations at p < 0.05")

    # Sector breakdown
    sector_results = run_sector_correlations(df)
    print_table(sector_results, "SECTOR-LEVEL Z-SCORED CORRELATIONS", "sector")

    # Sector significant findings
    sector_sig = sector_results[sector_results["significant"]]
    if len(sector_sig):
        print(f"\n{'='*65}")
        print(" SIGNIFICANT SECTOR-LEVEL FINDINGS")
        print(f"{'='*65}")
        for _, row in sector_sig.iterrows():
            direction = "↑ positive" if row["correlation"] > 0 else "↓ negative"
            print(f"  [{row['sector']}] {row['feature']} → "
                  f"{row['target']}: r={row['correlation']} "
                  f"p={row['pvalue']} {direction}")