import numpy as np
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from typing import List
from eva.utils import split_text_word

def compute_coherence(
    reference_corpus: List[str],
    vocab: List[str],
    top_words: List[List[str]],
    coherence_type: str = 'c_v',
    topn: int = 20
) -> float:
    split_reference_corpus = split_text_word(reference_corpus)
    dictionary = Dictionary(split_text_word(vocab))

    cm = CoherenceModel(
        texts=split_reference_corpus,
        dictionary=dictionary,
        topics=top_words,
        topn=topn,
        coherence=coherence_type,
    )
    score = cm.get_coherence()
    return score