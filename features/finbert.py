from transformers import pipeline
import torch

# Load once at module level so we don't reload the model on every call
# This will download ~500MB on first run — expected behavior
print("Loading FinBERT model (first run downloads ~500MB)...")
_classifier = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    tokenizer="ProsusAI/finbert",
    top_k=None,
    device=0 if torch.cuda.is_available() else -1,
    truncation=True,
    max_length=512,
)
print("FinBERT loaded.")


def chunk_text(text: str, max_words: int = 400) -> list[str]:
    """
    Split text into chunks of max_words.
    FinBERT has a 512 token limit so we chunk and aggregate.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        if chunk:
            chunks.append(chunk)
    return chunks


def finbert_sentiment(text: str) -> dict:
    """
    Run FinBERT on text, chunking if necessary.
    Returns averaged positive, negative, neutral scores.
    """
    if not text or len(text.split()) < 20:
        return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}

    chunks = chunk_text(text)

    # batch all chunks in a single call instead of looping
    all_results = _classifier(chunks, batch_size=8)

    pos_scores, neg_scores, neu_scores = [], [], []
    for results in all_results:
        scores = {r["label"].lower(): r["score"] for r in results}
        pos_scores.append(scores.get("positive", 0.0))
        neg_scores.append(scores.get("negative", 0.0))
        neu_scores.append(scores.get("neutral", 0.0))

    return {
        "positive": round(sum(pos_scores) / len(pos_scores), 4),
        "negative": round(sum(neg_scores) / len(neg_scores), 4),
        "neutral": round(sum(neu_scores) / len(neu_scores), 4),
    }
