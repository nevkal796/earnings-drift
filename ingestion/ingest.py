import time
from ingestion.edgar import get_cik, get_filings, get_exhibit_text
from ingestion.loader import upsert_company, upsert_filing, insert_transcript

# Companies to ingest — ticker and display name
COMPANIES = [
    ("AAPL", "Apple Inc."),
    ("MSFT", "Microsoft Corporation"),
    ("META", "Meta Platforms Inc."),
    ("NVDA", "NVIDIA Corporation"),
    ("GOOGL", "Alphabet Inc."),
]

def ingest_company(ticker: str, name: str, limit: int = 20):
    print(f"\n{'='*50}")
    print(f"Ingesting {ticker}...")

    # Step 1: get CIK
    cik = get_cik(ticker)
    if not cik:
        print(f"  ✗ Could not find CIK for {ticker}")
        return
    print(f"  CIK: {cik}")

    # Step 2: save company to DB
    company_id = upsert_company(ticker=ticker, cik=cik, name=name)
    print(f"  DB company_id: {company_id}")

    # Step 3: get filings
    filings = get_filings(cik, form_type="8-K", limit=limit)
    print(f"  Found {len(filings)} 8-K filings")

    saved = 0
    skipped = 0

    for f in filings:
        # Step 4: save filing to DB
        filing_id = upsert_filing(
            company_id=company_id,
            accession=f["accession"],
            filed=f["filed"],
            period=f["period"],
            form_type=f["form"]
        )

        # Step 5: fetch exhibit text
        text = get_exhibit_text(cik, f["accession"])

        if text:
            # Step 6: save transcript to DB
            insert_transcript(filing_id=filing_id, raw_text=text)
            saved += 1
            print(f"  ✓ {f['filed']} — {len(text):,} chars saved")
        else:
            skipped += 1
            print(f"  ✗ {f['filed']} — no EX-99.1 found")

        time.sleep(0.12)

    print(f"\n  Done: {saved} transcripts saved, {skipped} skipped")

if __name__ == "__main__":
    for ticker, name in COMPANIES:
        ingest_company(ticker, name, limit=20)
    print("\n\nAll companies ingested.")