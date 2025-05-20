import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class ContentBasedRecommender:
    def __init__(self, data):
        # Use unique movies for TF-IDF
        self.movies = data[['movieId', 'title', 'genres']].drop_duplicates().copy()
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.indices = None
        self._prepare()

    def _prepare(self):
        self.movies['genres'] = self.movies['genres'].fillna('')
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.movies['genres'])
        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)
        self.indices = pd.Series(self.movies.index, index=self.movies['title']).drop_duplicates()

    def recommend(self, title, top_n=10):
        title = title.lower()
        if title not in self.indices:
            return pd.DataFrame()
        idx = self.indices[title]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        movie_indices = [i[0] for i in sim_scores]
        return self.movies.iloc[movie_indices][['movieId', 'title', 'genres']]