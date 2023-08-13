import pandas as pd
import numpy as np
import gensim
from gensim import corpora
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from pprint import pprint


'''
word2vec 사용 콘텐츠 기반 추천 모델
'''
class Word2vecRecModel():

    def __init__(self, user_dataset : pd.DataFrame, news_dataset: pd.DataFrame, new_data: dict, **kwargs):

        self.user = user_dataset
        self.news = news_dataset
        self.new_data = new_data
        self.num_rec = kwargs.get("num_rec", 3)

    def recommend(self, ) -> str:
        
        all_data = []
        all_data.extend([" ".join(map(str, row)) for _, row in self.user.iterrows()])
        all_data.extend(self.news["title"])
        all_data.extend(self.news["content"])

        processed_data = [simple_preprocess(data) for data in all_data]

        dictionary = corpora.Dictionary(processed_data)
        corpus = [dictionary.doc2bow(text) for text in processed_data]

        word2vec_model = Word2Vec(sentences=processed_data, vector_size=100, window=5, min_count=1, workers=4, sg=1)

        def word2vec_based_recommendation(user_profile, model, num_recommendations=self.num_rec):
            user_profile_text = simple_preprocess(" ".join(map(str, self.new_data.values())))
            user_profile_vector = np.mean([model.wv[word] for word in user_profile_text if word in model.wv], axis=0)
            news_vectors = [np.mean([model.wv[word] for word in simple_preprocess(title) if word in model.wv], axis=0) for title in self.news["title"]]
            similarities = [np.dot(user_profile_vector, news_vector) / (np.linalg.norm(user_profile_vector) * np.linalg.norm(news_vector)) for news_vector in news_vectors]
            top_indices = np.argsort(similarities)[::-1][:num_recommendations]
            recommended_news = self.news.loc[top_indices, 'title'].tolist()
            return recommended_news

        recommended_news = word2vec_based_recommendation(self.new_data, word2vec_model)
        return recommended_news