import re


def count_syllables(word: str) -> int:
    """Approximate syllable count for a word."""
    word = word.lower().strip(".,;:!?\"'")
    if len(word) <= 3:
        return 1
    word = re.sub(r"(es|ed|e)$", "", word)
    word = re.sub(r"le$", "l", word)
    vowels = re.findall(r"[aeiouy]+", word)
    count = len(vowels)
    return max(1, count)


def is_complex_word(word: str) -> bool:
    """A complex word has 3 or more syllables."""
    return count_syllables(word) >= 3


def gunning_fog(text: str) -> float:
    """
    Gunning Fog Index — measures text complexity.
    Score 12 = high school senior level.
    Score 16 = college graduate level.
    Executives who are evasive tend to use more complex language.
    Formula: 0.4 * ((words/sentences) + 100 * (complex_words/words))
    """
    if not text:
        return 0.0

    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if not sentences:
        return 0.0

    words = text.split()
    if not words:
        return 0.0

    complex_words = sum(1 for w in words if is_complex_word(w))
    avg_sentence_length = len(words) / len(sentences)
    complex_word_ratio = complex_words / len(words)

    fog = 0.4 * (avg_sentence_length + 100 * complex_word_ratio)
    return round(fog, 4)