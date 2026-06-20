from features.hedging import hedging_score, uncertainty_score, specificity_score
from features.fog import gunning_fog
from features.finbert import finbert_sentiment
from ingestion.loader import get_connection


def compute_features(transcript_id: int, text: str) -> dict:
    """Compute all linguistic features for a transcript."""
    hedging = hedging_score(text)
    uncertainty = uncertainty_score(text)
    specificity = specificity_score(text)
    fog = gunning_fog(text)
    sentiment = finbert_sentiment(text)

    return {
        "transcript_id": transcript_id,
        "hedging_score": hedging,
        "uncertainty_score": uncertainty,
        "specificity_score": specificity,
        "fog_index": fog,
        "finbert_positive": sentiment["positive"],
        "finbert_negative": sentiment["negative"],
        "finbert_neutral": sentiment["neutral"],
    }


def save_features(features: dict):
    """Save computed features to the database."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO linguistic_features (
            transcript_id, hedging_score, uncertainty_score,
            specificity_score, fog_index,
            finbert_positive, finbert_negative, finbert_neutral
        ) VALUES (
            %(transcript_id)s, %(hedging_score)s, %(uncertainty_score)s,
            %(specificity_score)s, %(fog_index)s,
            %(finbert_positive)s, %(finbert_negative)s, %(finbert_neutral)s
        )
        ON CONFLICT (transcript_id) DO UPDATE SET
            hedging_score = EXCLUDED.hedging_score,
            uncertainty_score = EXCLUDED.uncertainty_score,
            specificity_score = EXCLUDED.specificity_score,
            fog_index = EXCLUDED.fog_index,
            finbert_positive = EXCLUDED.finbert_positive,
            finbert_negative = EXCLUDED.finbert_negative,
            finbert_neutral = EXCLUDED.finbert_neutral,
            computed_at = NOW()
    """, features)
    conn.commit()
    cur.close()
    conn.close()


def run_feature_pipeline():
    """Run feature extraction on all cleaned transcripts."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT t.id, t.prepared_remarks, c.ticker, f.filed_at
        FROM transcripts t
        JOIN filings f ON f.id = t.filing_id
        JOIN companies c ON c.id = f.company_id
        LEFT JOIN linguistic_features lf ON lf.transcript_id = t.id
        WHERE t.prepared_remarks IS NOT NULL
        AND lf.id IS NULL
        ORDER BY c.ticker, f.filed_at
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    print(f"Computing features for {len(rows)} transcripts...\n")

    for i, (transcript_id, text, ticker, filed_at) in enumerate(rows):
        if not text:
            print(f"  [{ticker}] {filed_at} — skipping, no prepared remarks")
            continue

        print(f"  [{i+1}/{len(rows)}] {ticker} {filed_at}...", end=" ", flush=True)
        features = compute_features(transcript_id, text)
        save_features(features)
        print(f"hedging={features['hedging_score']} "
              f"fog={features['fog_index']} "
              f"pos={features['finbert_positive']} "
              f"neg={features['finbert_negative']}")

    print(f"\nDone. Features saved for {len(rows)} transcripts.")


if __name__ == "__main__":
    run_feature_pipeline()