# collaborative.py

from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import pandas as pd

class CollaborativeRecommender:
    def __init__(self, df):
        self.df = df[['userId', 'movieId', 'rating']]
        self.model = None
        self.trainset = None
        self.testset = None
        self._prepare()

    def _prepare(self):
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(self.df, reader)
        self.trainset, self.testset = train_test_split(data, test_size=0.2, random_state=42)

        self.model = SVD()
        self.model.fit(self.trainset)

    def predict_rating(self, user_id, movie_id):
        pred = self.model.predict(user_id, movie_id)
        return pred.est

    def recommend(self, user_id, unseen_movies, top_n=10):
        predictions = [self.model.predict(user_id, movie_id) for movie_id in unseen_movies]
        predictions.sort(key=lambda x: x.est, reverse=True)
        top_preds = predictions[:top_n]
        return [(pred.iid, pred.est) for pred in top_preds]

    def evaluate(self):
        predictions = self.model.test(self.testset)
        rmse = accuracy.rmse(predictions, verbose=False)
        mae = accuracy.mae(predictions, verbose=False)
        return rmse, mae