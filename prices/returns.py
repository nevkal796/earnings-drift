import yfinance as yf
import time
from datetime import datetime, timedelta
from ingestion.loader import get_connection


def get_returns(ticker: str, filing_date: str) -> dict | None:
    """
    Fetch stock returns at 30, 60, and 90 days after a filing date.
    Returns None if price data is unavailable.
    """
    try:
        start = datetime.strptime(filing_date, "%Y-%m-%d")
        end = start + timedelta(days=100)

        stock = yf.Ticker(ticker)
        hist = stock.history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d")
        )

        if len(hist) < 5:
            return None

        filing_close = hist.iloc[0]["Close"]

        def safe_return(trading_days: int) -> float | None:
            idx = min(trading_days, len(hist) - 1)
            if idx < 1:
                return None
            return round(
                (hist.iloc[idx]["Close"] - filing_close) / filing_close, 6
            )

        return {
            "filing_date": filing_date,
            "close_price": round(float(filing_close), 4),
            "return_30d": safe_return(21),   # ~21 trading days = 30 calendar days
            "return_60d": safe_return(42),
            "return_90d": safe_return(63),
        }

    except Exception as e:
        print(f"    Error fetching {ticker} on {filing_date}: {e}")
        return None


def save_price_snapshot(company_id: int, data: dict):
    """Save a price snapshot to the database."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO price_snapshots (
            company_id, filing_date, close_price,
            return_30d, return_60d, return_90d
        ) VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (company_id, filing_date) DO UPDATE SET
            close_price = EXCLUDED.close_price,
            return_30d = EXCLUDED.return_30d,
            return_60d = EXCLUDED.return_60d,
            return_90d = EXCLUDED.return_90d
    """, (
        int(company_id),
        data["filing_date"],
        float(data["close_price"]),
        float(data["return_30d"]) if data["return_30d"] is not None else None,
        float(data["return_60d"]) if data["return_60d"] is not None else None,
        float(data["return_90d"]) if data["return_90d"] is not None else None,
    ))
    conn.commit()
    cur.close()
    conn.close()


def run_price_pipeline():
    """Fetch price data for all filings in the database."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT c.id, c.ticker, f.filed_at
        FROM companies c
        JOIN filings f ON f.company_id = c.id
        JOIN transcripts t ON t.filing_id = f.id
        JOIN linguistic_features lf ON lf.transcript_id = t.id
        LEFT JOIN price_snapshots ps 
            ON ps.company_id = c.id 
            AND ps.filing_date = f.filed_at
        WHERE ps.id IS NULL
        AND f.filed_at IS NOT NULL
        ORDER BY c.ticker, f.filed_at
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    print(f"Fetching price data for {len(rows)} filings...\n")

    saved = 0
    skipped = 0

    for company_id, ticker, filed_at in rows:
        filing_date = str(filed_at)
        print(f"  {ticker} {filing_date}...", end=" ", flush=True)

        data = get_returns(ticker, filing_date)

        if data:
            save_price_snapshot(company_id, data)
            r30 = f"{data['return_30d']*100:.2f}%" if data['return_30d'] else "N/A"
            r90 = f"{data['return_90d']*100:.2f}%" if data['return_90d'] else "N/A"
            print(f"✓ close=${data['close_price']} 30d={r30} 90d={r90}")
            saved += 1
        else:
            print("✗ no price data")
            skipped += 1

        time.sleep(0.3)  # be polite to Yahoo Finance

    print(f"\nDone. {saved} price snapshots saved, {skipped} skipped.")


if __name__ == "__main__":
    run_price_pipeline()