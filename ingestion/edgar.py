import requests
import time
import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup

load_dotenv()

HEADERS = {"User-Agent": os.getenv("SEC_USER_AGENT")}
BASE = "https://data.sec.gov"
ARCHIVE = "https://www.sec.gov/Archives/edgar/data"

def get_cik(ticker: str) -> str | None:
    url = f"{BASE}/submissions/CIK{ticker}.json"
    # CIK lookup via company tickers JSON
    r = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=HEADERS
    )
    data = r.json()
    for entry in data.values():
        if entry["ticker"].upper() == ticker.upper():
            return str(entry["cik_str"]).zfill(10)
    return None

def get_filings(cik: str, form_type="8-K", limit=40) -> list[dict]:
    url = f"{BASE}/submissions/CIK{cik}.json"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    data = r.json()
    recent = data["filings"]["recent"]
    results = []
    for i, form in enumerate(recent["form"]):
        if form == form_type:
            results.append({
                "accession": recent["accessionNumber"][i],
                "filed": recent["filingDate"][i],
                "period": recent["reportDate"][i] if recent["reportDate"][i] else None,
                "form": form
            })
        if len(results) >= limit:
            break
    time.sleep(0.12)  # stay under 10 req/sec
    return results

def get_exhibit_text(cik: str, accession: str) -> str | None:
    """Fetch exhibit text from an 8-K filing."""
    acc_clean = accession.replace("-", "")
    cik_int = int(cik)
    index_url = f"{ARCHIVE}/{cik_int}/{acc_clean}/{accession}-index.htm"

    r = requests.get(index_url, headers=HEADERS)
    time.sleep(0.12)

    if r.status_code != 200:
        return None

    soup = BeautifulSoup(r.text, "html.parser")

    # Strategy 1: find EX-99.1 by checking cells[1] for exhibit type
    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) >= 3:
            exhibit_type = cells[1].text.strip()
            if "EX-99.1" in exhibit_type or "99.1" in exhibit_type:
                link = cells[2].find("a")
                if not link:
                    link = cells[1].find("a")
                if link:
                    href = link["href"].split()[0]  # strip iXBRL suffix if present
                    doc_url = f"https://www.sec.gov{href}"
                    doc_r = requests.get(doc_url, headers=HEADERS)
                    time.sleep(0.12)
                    doc_soup = BeautifulSoup(doc_r.text, "html.parser")
                    text = doc_soup.get_text(separator=" ", strip=True)
                    if len(text) > 500:
                        return text

    # Strategy 2: any .htm that isn't the main 8-K or XBRL
    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) >= 3:
            exhibit_type = cells[1].text.strip()
            link = cells[2].find("a")
            if link:
                href = link["href"].split()[0]
                if (href.endswith(".htm") and
                    "EX-101" not in exhibit_type and
                    exhibit_type not in ["8-K", "GRAPHIC", "XML", ""]):
                    doc_url = f"https://www.sec.gov{href}"
                    doc_r = requests.get(doc_url, headers=HEADERS)
                    time.sleep(0.12)
                    doc_soup = BeautifulSoup(doc_r.text, "html.parser")
                    text = doc_soup.get_text(separator=" ", strip=True)
                    if len(text) > 500:
                        return text

    return None