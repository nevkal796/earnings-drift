import re

# Patterns that signal the start of Q&A section
QA_SIGNALS = [
    r"question.and.answer",
    r"q\s*&\s*a",
    r"open.{0,20}floor.{0,20}questions",
    r"operator.*please.{0,30}questions",
    r"we will now.{0,30}question",
    r"take.{0,20}questions",
    r"first question",
]

# Boilerplate patterns to strip
JUNK_PATTERNS = [
    r"safe harbor.{0,500}forward.looking",
    r"forward.looking statements.{0,1000}actual results",
    r"this (transcript|document) (is|has been) (provided|prepared).{0,300}",
    r"©.{0,100}all rights reserved",
    r"page \d+ of \d+",
    r"\s{3,}",  # excessive whitespace
]


def clean_text(text: str) -> str:
    """Remove junk, normalize whitespace."""
    if not text:
        return ""

    # normalize unicode
    text = text.encode("ascii", "ignore").decode("ascii")

    # remove junk patterns
    for pattern in JUNK_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE | re.DOTALL)

    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def split_sections(text: str) -> tuple[str, str]:
    """
    Split transcript into prepared remarks and Q&A section.
    Returns (prepared_remarks, qa_section).
    If no Q&A found, returns (full_text, "").
    """
    text_lower = text.lower()
    split_pos = None

    for pattern in QA_SIGNALS:
        match = re.search(pattern, text_lower)
        if match:
            # take the earliest match
            if split_pos is None or match.start() < split_pos:
                split_pos = match.start()

    if split_pos is None:
        # no Q&A section detected
        return text.strip(), ""

    prepared = text[:split_pos].strip()
    qa = text[split_pos:].strip()

    return prepared, qa


def process_transcript(raw_text: str) -> dict:
    """
    Full cleaning pipeline for a raw transcript.
    Returns dict with cleaned fields and basic stats.
    """
    cleaned = clean_text(raw_text)
    prepared, qa = split_sections(cleaned)

    return {
        "cleaned_text": cleaned,
        "prepared_remarks": prepared,
        "qa_section": qa,
        "word_count": len(cleaned.split()),
        "prepared_word_count": len(prepared.split()),
        "qa_word_count": len(qa.split()),
        "has_qa": len(qa) > 100,
    }