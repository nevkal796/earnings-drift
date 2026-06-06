import pandas as pd
from scipy import stats
from ingestion.loader import get_connection


def load_dataset() -> pd.DataFrame:
    """
    Join all tables into a single analysis dataframe.
    Excludes outlier transcripts with abnormal feature scores.
    """
    conn = get_connection()
    query = """
        SELECT
            c.ticker,
            f.filed_at,
            lf.hedging_score,
            lf.uncertainty_score,
            lf.specificity_score,
            lf.fog_index,
            lf.finbert_positive,
            lf.finbert_negative,
            lf.finbert_neutral,
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
        AND ps.return_90d IS NOT NULL
        AND lf.hedging_score < 30
        AND lf.fog_index < 35
        AND t.word_count > 200
        ORDER BY c.ticker, f.filed_at
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def spearman_correlation(df: pd.DataFrame, feature: str, target: str):
    """Run Spearman correlation between a feature and a return target."""
    clean = df[[feature, target]].dropna()
    if len(clean) < 5:
        return None
    corr, pvalue = stats.spearmanr(clean[feature], clean[target])
    return {
        "feature": feature,
        "target": target,
        "n": len(clean),
        "correlation": round(corr, 4),
        "pvalue": round(pvalue, 4),
        "significant": pvalue < 0.05,
    }


def run_correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run Spearman correlations between all linguistic features
    and all return horizons. Returns a results dataframe.
    """
    features = [
        "hedging_score",
        "uncertainty_score",
        "specificity_score",
        "fog_index",
        "finbert_positive",
        "finbert_negative",
    ]
    targets = ["return_30d", "return_60d", "return_90d"]

    results = []
    for feature in features:
        for target in targets:
            result = spearman_correlation(df, feature, target)
            if result:
                results.append(result)

    return pd.DataFrame(results)


def per_company_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Run correlations broken down by company."""
    features = ["hedging_score", "finbert_positive", "finbert_negative", "fog_index"]
    targets = ["return_30d", "return_90d"]
    results = []

    for ticker in df["ticker"].unique():
        company_df = df[df["ticker"] == ticker]
        for feature in features:
            for target in targets:
                result = spearman_correlation(company_df, feature, target)
                if result:
                    result["ticker"] = ticker
                    results.append(result)

    return pd.DataFrame(results)


def print_results(results_df: pd.DataFrame, title: str):
    """Pretty print correlation results."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")
    print(f"{'Feature':<22} {'Target':<12} {'N':>4} {'Corr':>8} {'P-Value':>8} {'Sig':>5}")
    print(f"{'-'*60}")

    for _, row in results_df.iterrows():
        sig = "✓" if row["significant"] else ""
        ticker = f"[{row['ticker']}] " if "ticker" in row else ""
        print(
            f"{ticker}{row['feature']:<22} "
            f"{row['target']:<12} "
            f"{int(row['n']):>4} "
            f"{row['correlation']:>8.4f} "
            f"{row['pvalue']:>8.4f} "
            f"{sig:>5}"
        )


if __name__ == "__main__":
    print("Loading dataset...")
    df = load_dataset()
    print(f"Dataset: {len(df)} observations across {df['ticker'].nunique()} companies")
    print(f"Date range: {df['filed_at'].min()} to {df['filed_at'].max()}")
    print(f"\nDescriptive stats:")
    print(df[["hedging_score", "fog_index", "finbert_positive",
              "finbert_negative", "return_30d", "return_90d"]].describe().round(4))

    # Overall correlations
    overall = run_correlation_analysis(df)
    print_results(overall, "OVERALL CORRELATIONS — All Companies")

    # Per company
    per_company = per_company_correlations(df)
    print(f"\n{'='*60}")
    print(" PER-COMPANY CORRELATIONS")
    print(f"{'='*60}")
    for ticker in df["ticker"].unique():
        subset = per_company[per_company["ticker"] == ticker]
        print_results(subset, f"{ticker}")

    # Highlight significant findings
    significant = overall[overall["significant"] == True]
    if len(significant) > 0:
        print(f"\n{'='*60}")
        print(f" STATISTICALLY SIGNIFICANT FINDINGS (p < 0.05)")
        print(f"{'='*60}")
        for _, row in significant.iterrows():
            direction = "↑ positive" if row["correlation"] > 0 else "↓ negative"
            print(f"  {row['feature']} → {row['target']}: "
                  f"r={row['correlation']} p={row['pvalue']} {direction}")
    else:
        print("\n No statistically significant correlations at p < 0.05")
        print(" (Expected with n=60 — expand to more companies in week 2)")