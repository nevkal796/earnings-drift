from ingestion.loader import get_connection
from ingestion.cleaner import process_transcript

def clean_all_transcripts():
    conn = get_connection()
    cur = conn.cursor()

    # fetch all transcripts that haven't been cleaned yet
    cur.execute("""
        SELECT t.id, t.raw_text, c.ticker
        FROM transcripts t
        JOIN filings f ON f.id = t.filing_id
        JOIN companies c ON c.id = f.company_id
        WHERE t.cleaned_text IS NULL
        AND t.raw_text IS NOT NULL
    """)
    rows = cur.fetchall()
    print(f"Found {len(rows)} transcripts to clean\n")

    for transcript_id, raw_text, ticker in rows:
        result = process_transcript(raw_text)

        cur.execute("""
            UPDATE transcripts SET
                cleaned_text = %s,
                prepared_remarks = %s,
                qa_section = %s,
                word_count = %s,
                prepared_word_count = %s,
                qa_word_count = %s,
                has_qa = %s
            WHERE id = %s
        """, (
            result["cleaned_text"],
            result["prepared_remarks"],
            result["qa_section"],
            result["word_count"],
            result["prepared_word_count"],
            result["qa_word_count"],
            result["has_qa"],
            transcript_id
        ))

        qa_status = "✓ Q&A" if result["has_qa"] else "✗ no Q&A"
        print(f"  [{ticker}] transcript {transcript_id} — "
              f"{result['word_count']:,} words — {qa_status}")

    conn.commit()
    cur.close()
    conn.close()
    print(f"\nDone. {len(rows)} transcripts cleaned and updated.")

if __name__ == "__main__":
    clean_all_transcripts()