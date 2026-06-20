# Earnings Call Sentiment Drift: Linguistic Features as Predictors of Post-Filing Returns

**A research memo on the relationship between executive communication patterns in SEC 8-K filings and subsequent stock price movement**

---

## Abstract

This study investigates whether measurable linguistic features in executive earnings communications — specifically hedging frequency, uncertainty language, numeric specificity, readability complexity, and sentiment polarity — carry predictive signal about subsequent stock returns. Using a corpus of 281 SEC 8-K earnings press releases (exhibit 99.1) from 18 publicly traded companies across 6 sectors, spanning January 2024 through May 2026, we extract five linguistic features per filing and test their correlation with 30, 60, and 90-day forward returns using Spearman rank correlation.

We find that **numeric specificity** — the density of concrete figures, percentages, and guidance ranges in a filing — exhibits a statistically significant negative correlation with 30-day forward returns (r = -0.26, p < 0.001, n = 281) that strengthens rather than weakens as the corpus expands, and partially survives within-company normalization (r = -0.13, p = 0.027). This finding is counterintuitive: more precise communication predicts *lower* subsequent returns, consistent with a hypothesis that executives use specific figures to anchor expectations downward ahead of disappointing results, while more positive surprises are communicated with comparatively less numeric precision.

Other features — hedging frequency, uncertainty language, and sentiment polarity — show no significant relationship at the cross-company level but emerge as significant predictors *within specific companies and sectors*, indicating that linguistic baselines are company- and sector-specific rather than universal.

---

## 1. Research Question

Do executives signal forthcoming bad news through detectable shifts in communication style before the market fully prices it in? Prior research in accounting and finance (notably the development of the Loughran-McDonald financial sentiment dictionary) has established that general-purpose sentiment lexicons perform poorly on financial text, and that domain-specific linguistic features carry information not captured by reported financial metrics alone. This study extends that literature by testing five specific linguistic dimensions against near-term forward returns using a fully reproducible, open-data pipeline.

---

## 2. Data

### 2.1 Corpus construction

Filings were sourced directly from the SEC EDGAR full-text search API (`data.sec.gov`), targeting Form 8-K exhibit 99.1 documents — the standard format for quarterly earnings press releases. For each of 18 companies, up to 20 of the most recent 8-K filings were retrieved and the EX-99.1 exhibit text was extracted via HTML parsing of each filing's index page.

**Final corpus:** 281 filings with usable text, non-null 30-day forward returns, word count > 200, and feature values within plausible bounds (hedging score < 30, fog index < 35 — see Section 2.3 on outlier handling).

### 2.2 Company selection

Eighteen companies were selected across six sectors to test whether linguistic signals generalize across industries or are sector-specific:

| Sector | Companies |
|---|---|
| Technology | AAPL, MSFT, META, NVDA, GOOGL |
| Finance | JPM, GS, BAC |
| Healthcare | JNJ, UNH, PFE |
| Energy | XOM, CVX |
| Retail | WMT, COST |
| Industrial | CAT, BA |
| Consumer | KO |

### 2.3 Outlier handling

A small number of filings (n=2 in early testing) returned hedging scores an order of magnitude above the corpus norm (>50 vs a corpus mean of ~7) due to the EX-99.1 extraction occasionally capturing legal or governance documents rather than genuine earnings releases. These were excluded via a threshold filter (hedging_score < 30, fog_index < 35) rather than manual removal, to keep the filtering criteria reproducible and auditable.

### 2.4 Price data

Forward returns were computed using daily close prices from Yahoo Finance (`yfinance`), measured at approximately 21, 42, and 63 trading days following each filing date (corresponding to 30, 60, and 90 calendar days). Filings without at least 90 days of subsequent trading history at the time of analysis were excluded from 90-day return calculations.

---

## 3. Methodology

### 3.1 Linguistic features

**Hedging score** — frequency of epistemic hedging terms (*believe, could, may, approximately, potentially*) per 1,000 words, based on a hedging lexicon derived from the Loughran-McDonald financial sentiment dictionary categories.

**Uncertainty score** — frequency of explicit risk and instability language (*headwind, volatile, pressure, challenge*) per 1,000 words. Conceptually distinct from hedging: hedging measures the speaker's epistemic confidence, uncertainty measures references to external risk.

**Specificity score** — density of concrete numeric references (dollar figures, percentages, basis points, quarter/fiscal-year references) per 1,000 words, identified via regex pattern matching.

**Gunning Fog Index** — a standard readability metric (Gunning, 1952): `0.4 × (average sentence length + 100 × proportion of words with 3+ syllables)`. Higher scores indicate denser, harder-to-parse prose.

**FinBERT sentiment** — positive, negative, and neutral probability scores from `ProsusAI/finbert`, a BERT-base model fine-tuned on financial text (Araci, 2019). Long documents were chunked at 400-word segments (under FinBERT's 512-token limit) and scores were averaged across chunks.

### 3.2 Statistical approach

Spearman rank correlation was used throughout rather than Pearson correlation, for two reasons: (1) financial returns are heavy-tailed and contain outliers (single-quarter moves of 20%+ are common in this corpus) that distort linear correlation measures, and (2) there is no strong theoretical prior that the relationship between linguistic features and returns should be linear. Spearman correlation operates on ranks and is robust to both concerns.

Significance was assessed at the conventional α = 0.05 threshold. Given the number of feature-target pairs tested (6 features × 3 horizons = 18 tests at the corpus level alone), we note that no multiple-comparison correction (e.g., Bonferroni) was applied; the headline specificity finding (p < 0.001) survives even a conservative correction, but secondary findings should be read as exploratory rather than confirmatory.

### 3.3 Within-company normalization

Because baseline communication style plausibly differs by company and sector (a hedging score of 8 may be unremarkable for one company and elevated for another), features were additionally z-scored within each company — i.e., each filing's feature value was expressed as a deviation from that company's own historical mean, in units of that company's own standard deviation — before re-running the correlation analysis. This isolates *within-company* signal from *between-company* baseline differences.

---

## 4. Results

### 4.1 Corpus-wide correlations (raw features)

| Feature | Horizon | n | Spearman r | p-value | Significant |
|---|---|---|---|---|---|
| Specificity score | 30d | 281 | -0.2625 | <0.0001 | ✓ |
| Specificity score | 60d | 281 | -0.1735 | 0.0035 | ✓ |
| Specificity score | 90d | 281 | -0.1013 | 0.0902 | |
| Hedging score | 30d | 281 | -0.1095 | 0.0669 | |
| Uncertainty score | 90d | 281 | -0.0785 | 0.1897 | |
| Fog index | all horizons | 281 | ≈ 0 | n.s. | |
| FinBERT positive | all horizons | 281 | -0.06 to -0.10 | n.s. | |
| FinBERT negative | all horizons | 281 | -0.08 to 0.03 | n.s. | |

Specificity score is the only feature with a significant cross-company relationship, and it strengthened rather than weakened when the corpus was expanded from an initial 56 observations (5 companies) to the full 281 (18 companies) — evidence against the finding being a small-sample artifact.

### 4.2 Within-company z-scored correlations

| Feature (z-scored) | Horizon | n | Spearman r | p-value | Significant |
|---|---|---|---|---|---|
| Specificity score | 30d | 281 | -0.1319 | 0.0270 | ✓ |
| Fog index | 30d | 281 | -0.1247 | 0.0367 | ✓ |

The specificity effect persists, at reduced magnitude, after removing between-company baseline differences — indicating the signal exists both *across* companies and *within* a given company's own history over time. The fog index effect, which was not significant in raw form, becomes significant after normalization, suggesting it is masked by between-company differences in baseline writing complexity until those are controlled for.

### 4.3 Sector- and company-level findings

| Group | Feature | Horizon | n | Spearman r | p-value |
|---|---|---|---|---|---|
| AAPL | Hedging score | 30d | 11 | -0.70 | 0.017 |
| AAPL | FinBERT positive | 30d | 11 | +0.61 | 0.048 |
| GOOGL | Fog index | 30d | 9 | -0.77 | 0.016 |
| Technology (sector) | Fog index (z) | 30d | — | -0.25 | 0.006 |
| Industrial (sector) | Specificity score (z) | 30d | — | -0.43 | 0.015 |
| Industrial (sector) | FinBERT negative (z) | 30d | — | -0.45 | 0.010 |

These results indicate that hedging and sentiment features, while not significant at the full-corpus level, are meaningful predictors *within specific companies and sectors* — consistent with the hypothesis that linguistic baselines are not universal. Apple's normally direct, low-hedging communication style makes deviations from that baseline an informative signal; this would not be detectable in a model trained only on cross-company pooled data without company-level context.

---

## 5. Interpretation

The central finding — that numeric specificity negatively predicts near-term returns — runs counter to a naive prior that more concrete, confident-sounding communication should be associated with better outcomes. A plausible mechanism: when executives anticipate disappointing results, they may use specific figures to set a clear, defensible benchmark ("we expect approximately 3% growth") precisely because vague language would leave more room for the market to assume optimistic outcomes. Conversely, when results are genuinely strong, less defensive precision is needed.

This is consistent with — though not proof of — a broader finding in the accounting literature that managers exhibit asymmetric disclosure behavior around bad news (e.g., Kothari, Shu, and Wysocki, 2009, on the gradual versus abrupt release of bad versus good news). This study does not establish causality and the mechanism described above is a hypothesis for the observed correlation, not a confirmed explanation.

The fact that hedging, uncertainty, and sentiment features are *not* significant cross-sectionally but *are* significant within individual companies and sectors is itself an important finding: it implies that any production application of this kind of signal would need to be built on company-specific or sector-specific baselines rather than pooled, cross-company models.

---

## 6. Limitations

- **Sample size.** n = 281 at the corpus level is modest by financial econometrics standards; per-company subsamples (9–15 observations) are too small for robust standalone inference and are reported as exploratory.
- **No out-of-sample validation.** All correlations reported here are in-sample. The relationships have not been tested on a held-out time period or held-out set of companies, and should not be interpreted as a validated trading signal.
- **No multiple-comparison correction applied** across the full set of feature × horizon × grouping tests; the headline finding is robust to standard corrections, secondary findings are not adjusted and should be treated cautiously.
- **EX-99.1 coverage gaps.** Not all 8-K filings contain a usable EX-99.1 exhibit (governance and material-event 8-Ks were excluded by design), and extraction occasionally pulled non-earnings text, addressed via the outlier filter described in Section 2.3 rather than manual verification of every filing.
- **Press releases only, not full call transcripts.** This corpus uses 8-K exhibit text (prepared, legally reviewed press releases) rather than live earnings call transcripts including analyst Q&A. Executive language under live questioning may behave differently than scripted release language; this is a natural extension (see Section 7).
- **Survivorship and selection.** All 18 companies are large, established, currently-listed firms. Findings may not generalize to small-cap or distressed companies, which were not included in this corpus.

---

## 7. Future Work

- Expand the corpus to 50+ companies for greater statistical power, particularly for sector-level subgroup analysis.
- Incorporate full earnings call transcripts (prepared remarks plus analyst Q&A) to test whether the specificity effect holds, strengthens, or reverses under live questioning.
- Conduct genuine out-of-sample testing: fit relationships on data through a cutoff date and test predictive power on filings after that date.
- Apply topic modeling (e.g., BERTopic) to track the emergence and frequency of specific themes (supply chain, macro conditions, AI investment) over time and test whether topic shifts carry independent predictive signal.
- Test whether a combined feature (e.g., specificity score interacted with sector) outperforms any single feature in out-of-sample prediction.

---

## References

- Araci, D. (2019). *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models.* arXiv:1908.10063.
- Gunning, R. (1952). *The Technique of Clear Writing.* McGraw-Hill.
- Kothari, S.P., Shu, S., & Wysocki, P.D. (2009). *Do Managers Withhold Bad News?* Journal of Accounting Research, 47(1), 241–276.
- Loughran, T., & McDonald, B. (2011). *When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks.* The Journal of Finance, 66(1), 35–65.
- U.S. Securities and Exchange Commission, EDGAR full-text search system: https://www.sec.gov/edgar

---

## Reproducibility

All code, raw feature extraction logic, and the analysis pipeline used to produce these results are available in this repository. See [`README.md`](./README.md) for setup instructions and [`analysis/correlations.py`](./analysis/correlations.py) and [`analysis/zscore.py`](./analysis/zscore.py) for the exact statistical procedures used to generate the tables in this memo.
