def compute_diversity(top_words: list[list[str]]) -> float:
    num_words = 0
    word_set = set()
    for topic in top_words:
        num_words += len(topic)
        word_set.update(topic)

    return len(word_set) / num_words if num_words > 0 else 0.0