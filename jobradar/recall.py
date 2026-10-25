"""Stage 1: cheap recall. Score every filtered job against each resume variant,
keep the max. Only the top_k go to the (paid) Claude rerank.

tfidf: zero dependencies beyond sklearn, weak on synonyms ("CV" vs "computer vision").
embeddings: pip install sentence-transformers; better recall, slower first run.
Measure which one loses fewer of your positive labels with `jobradar eval`.
"""
from __future__ import annotations

import numpy as np


def score(job_texts: list[str], resume_texts: list[str], backend: str = "tfidf") -> np.ndarray:
    if not job_texts:
        return np.zeros(0)
    if backend == "embeddings":
        return _embed(job_texts, resume_texts)
    return _tfidf(job_texts, resume_texts)


def _tfidf(job_texts, resume_texts):
    return tfidf_matrix(job_texts, resume_texts).max(axis=1)


def tfidf_matrix(job_texts, resume_texts) -> np.ndarray:
    """Cosine of every job against every resume (jobs x resumes), on a vocabulary fit to both."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    m = vec.fit_transform(job_texts + resume_texts)
    jobs, resumes = m[: len(job_texts)], m[len(job_texts):]
    return cosine_similarity(jobs, resumes)


def _embed(job_texts, resume_texts, model_name="BAAI/bge-small-en-v1.5"):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    j = model.encode(job_texts, normalize_embeddings=True, batch_size=32)
    r = model.encode(resume_texts, normalize_embeddings=True)
    return (j @ r.T).max(axis=1)
