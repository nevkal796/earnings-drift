import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    """Create and return a database connection."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )

def upsert_company(ticker: str, cik: str, name: str = None, sector: str = None) -> int:
    """Insert a company if it doesn't exist, return its id."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO companies (ticker, cik, name, sector)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (ticker) DO UPDATE
            SET cik = EXCLUDED.cik,
                name = EXCLUDED.name,
                sector = EXCLUDED.sector
        RETURNING id
    """, (ticker.upper(), cik, name, sector))
    company_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return company_id
    
def upsert_filing(company_id: int, accession: str, filed: str, period: str, form_type: str) -> int:
    """Insert a filing if it doesn't exist, return its id."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO filings (company_id, accession_number, filed_at, period_of_report, form_type)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (accession_number) DO NOTHING
        RETURNING id
    """, (company_id, accession, filed, period, form_type))
    row = cur.fetchone()
    if row is None:
        # already existed, fetch its id
        cur.execute("SELECT id FROM filings WHERE accession_number = %s", (accession,))
        row = cur.fetchone()
    filing_id = row[0]
    conn.commit()
    cur.close()
    conn.close()
    return filing_id

def insert_transcript(filing_id: int, raw_text: str) -> int:
    """Insert raw transcript text, return its id."""
    conn = get_connection()
    cur = conn.cursor()
    word_count = len(raw_text.split())
    cur.execute("""
        INSERT INTO transcripts (filing_id, raw_text, word_count)
        VALUES (%s, %s, %s)
        ON CONFLICT DO NOTHING
        RETURNING id
    """, (filing_id, raw_text, word_count))
    row = cur.fetchone()
    if row is None:
        cur.execute("SELECT id FROM transcripts WHERE filing_id = %s", (filing_id,))
        row = cur.fetchone()
    transcript_id = row[0]
    conn.commit()
    cur.close()
    conn.close()
    return transcript_id