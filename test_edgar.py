from ingestion.edgar import get_cik, get_filings, get_exhibit_text

ticker = "AAPL"
print(f"Looking up CIK for {ticker}...")
cik = get_cik(ticker)
print(f"CIK: {cik}\n")

print("Fetching recent 8-K filings...")
filings = get_filings(cik, form_type="8-K", limit=5)

for f in filings:
    print(f"\nTrying {f['accession']} filed {f['filed']}...")
    text = get_exhibit_text(cik, f["accession"])
    if text:
        print(f"✓ Got {len(text)} characters")
        print(f"\nPreview:\n{text[:600]}")
        break
    else:
        print("✗ No EX-99.1 found, trying next...")