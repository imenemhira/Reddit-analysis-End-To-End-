import logging

from sklearn.feature_extraction.text import TfidfVectorizer

logger=logging.getLogger(__name__)
def create_tfidf_vectorizer():
    logger.info('creating tf-idf vectorizer')
    tfidf=TfidfVectorizer(
        max_features=5000,
        ngram_range=(1,2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    return tfidf