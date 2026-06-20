# Earnings Call Sentiment Drift Tracker

**Live dashboard:** https://your-railway-url.up.railway.app

A research pipeline that extracts linguistic features from SEC 8-K earnings filings and correlates them with subsequent stock price returns. Built to investigate whether executive communication patterns — hedging frequency, sentiment, readability complexity — carry predictive signal about near-term price movement.

---

## Research Question

Do measurable changes in executive language in earnings press releases predict subsequent stock returns? Specifically, do executives use more hedging language, more complex prose, or more negative sentiment in quarters that precede negative price movement?

---

## Findings

Analysis of **56 observations** across **5 companies** (AAPL, GOOGL, META, MSFT, NVDA) from **January 2024 – May 2026**:

| Finding | Correlation | P-Value | Interpretation |
|---|---|---|---|
| Specificity Score → 30d Return | r = -0.36 | p = 0.007 ✓ | More precise language precedes lower returns — executives anchor expectations downward with specific numbers |
| AAPL Hedging → 30d Return | r = -0.70 | p = 0.017 ✓ | Apple's normally direct communication style: elevated hedging is a strong negative signal |
| AAPL FinBERT Positive → 30d Return | r = +0.61 | p = 0.048 ✓ | Positive sentiment in Apple filings predicts positive near-term returns |
| GOOGL Fog Index → 30d Return | r = -0.77 | p = 0.016 ✓ | Google's filings become linguistically denser before negative price moves |

**Key insight:** Linguistic signals are company-specific, not universal. Each executive team has a distinct communication baseline — deviations from that baseline carry signal, raw scores do not. This motivates z-score normalization as a natural extension.

**Honest limitations:** n=56 is insufficient for robust cross-sectional inference. Per-company samples (9–14 observations) are underpowered for generalization. Findings are directionally interesting but should be validated on a larger corpus before drawing strong conclusions.

---

## Architecture
SEC EDGAR API → ingestion pipeline → PostgreSQL
↓
NLP feature extraction
(FinBERT + hedging lexicon
+ Gunning Fog + specificity)
↓
Yahoo Finance price data
↓
Spearman correlation analysis
↓
Plotly Dash dashboard

---

## Linguistic Features

**Hedging Score** — frequency of epistemic hedging terms (may, could, believe, approximately, potentially) per 1000 words. Based on the Loughran-McDonald financial sentiment lexicon.

**Uncertainty Score** — frequency of explicit uncertainty and risk language (volatile, headwind, pressure, challenge) per 1000 words.

**Specificity Score** — density of concrete numeric references (revenue figures, percentages, guidance ranges) per 1000 words. Counterintuitively negative — see findings.

**Gunning Fog Index** — readability metric measuring average sentence length and proportion of complex words (3+ syllables). Higher scores indicate denser, harder-to-parse prose.

**FinBERT Sentiment** — positive/negative/neutral scores from ProsusAI/finbert, a BERT model fine-tuned on financial text. Chunked at 400 words and averaged across the full document.

---

## Stack

- **Ingestion:** Python, SEC EDGAR API, BeautifulSoup
- **Storage:** PostgreSQL with a normalized relational schema
- **NLP:** Hugging Face Transformers (FinBERT), custom lexicon-based scorers
- **Price data:** yfinance
- **Analysis:** pandas, scipy (Spearman correlation)
- **Dashboard:** Plotly Dash

---

## Setup

**Prerequisites:** Python 3.11+, PostgreSQL 18

```bash
git clone https://github.com/YOURUSERNAME/earnings-drift.git
cd earnings-drift
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file:
DB_HOST=localhost
DB_PORT=5432
DB_NAME=earnings_drift
DB_USER=postgres
DB_PASSWORD=your_password
SEC_USER_AGENT=your_email@gmail.com

Create the database and run schema:

```bash
psql -U postgres -c "CREATE DATABASE earnings_drift;"
psql -U postgres -d earnings_drift -f db/schema.sql
```

Run the full pipeline:

```bash
python -m ingestion.ingest       # fetch SEC filings
python -m ingestion.clean_all    # clean and segment text
python -m features.pipeline      # extract linguistic features
python -m prices.returns         # fetch price data
python -m analysis.correlations  # run correlation analysis
python -m dashboard.app          # launch dashboard at localhost:8050
```

---

## Extending This Project

- **Expand corpus** — increase to 50+ companies across multiple sectors for statistical power
- **Z-score normalization** — normalize features against each company's historical baseline before cross-sectional analysis
- **Topic modeling** — BERTopic on the full corpus to identify recurring themes (supply chain, hiring, macro) and track their frequency over time
- **Full transcript ingestion** — add Motley Fool transcript scraping for Q&A section analysis
- **Sector comparison** — does the hedging signal hold in tech but not in energy? Are the signals sector-specific?

---

## Data Sources

- SEC EDGAR public API (no key required)
- Yahoo Finance via yfinance
- ProsusAI/finbert via Hugging Face
- Loughran-McDonald Financial Sentiment Dictionary (lexicon basis)