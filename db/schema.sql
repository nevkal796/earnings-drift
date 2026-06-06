CREATE TABLE IF NOT EXISTS companies (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) UNIQUE NOT NULL,
    cik VARCHAR(20) UNIQUE NOT NULL,
    name VARCHAR(255),
    sector VARCHAR(100),
    sic_code VARCHAR(10)
);

CREATE TABLE IF NOT EXISTS filings (
    id SERIAL PRIMARY KEY,
    company_id INTEGER REFERENCES companies(id),
    filed_at DATE,
    period_of_report DATE,
    accession_number VARCHAR(50) UNIQUE NOT NULL,
    form_type VARCHAR(20)
);

CREATE TABLE IF NOT EXISTS transcripts (
    id SERIAL PRIMARY KEY,
    filing_id INTEGER REFERENCES filings(id),
    raw_text TEXT,
    prepared_remarks TEXT,
    qa_section TEXT,
    word_count INTEGER,
    ingested_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS linguistic_features (
    id SERIAL PRIMARY KEY,
    transcript_id INTEGER REFERENCES transcripts(id) UNIQUE,
    hedging_score FLOAT,
    uncertainty_score FLOAT,
    specificity_score FLOAT,
    fog_index FLOAT,
    finbert_positive FLOAT,
    finbert_negative FLOAT,
    finbert_neutral FLOAT,
    topic_vector JSONB,
    computed_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS price_snapshots (
    id SERIAL PRIMARY KEY,
    company_id INTEGER REFERENCES companies(id),
    filing_date DATE,
    close_price FLOAT,
    return_30d FLOAT,
    return_60d FLOAT,
    return_90d FLOAT,
    UNIQUE(company_id, filing_date)
);