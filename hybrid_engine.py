# hybrid_engine.py

import pandas as pd

class HybridRecommender:
    def __init__(self, content_model, collaborative_model, data, weight_cb=0.5, weight_cf=0.5):
        self.content_model = content_model
        self.collaborative_model = collaborative_model
        self.data = data
        self.weight_cb = weight_cb
        self.weight_cf = weight_cf

    def recommend(self, user_id, input_title, top_n=10):
        # Content-based recommendations
        cb_recs = self.content_model.recommend(input_title, top_n=50)
        if cb_recs is None or cb_recs.empty:
            return pd.DataFrame()

        cb_recs = cb_recs.copy()
        cb_recs['cb_score'] = range(len(cb_recs), 0, -1)  # simple ranking-based score
        cb_recs['cb_score'] = cb_recs['cb_score'] / cb_recs['cb_score'].max()  # normalize

        # Collaborative predictions for those movies
        cb_recs['cf_score'] = cb_recs['movieId'].apply(
            lambda movie_id: self.collaborative_model.predict_rating(user_id, movie_id)
        )
        cb_recs['cf_score'] = cb_recs['cf_score'] / 5.0  # normalize

        # Hybrid score
        cb_recs['hybrid_score'] = (
            self.weight_cb * cb_recs['cb_score'] +
            self.weight_cf * cb_recs['cf_score']
        )

        hybrid_recs = cb_recs.sort_values(by='hybrid_score', ascending=False).head(top_n)
        return hybrid_recs[['movieId', 'title', 'genres', 'hybrid_score']]