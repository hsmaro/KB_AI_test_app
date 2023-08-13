import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity


'''
PLM 기반의 모델
사전 학습 모델을 통한 기사 추천
'''

class PLMRecModel():

    def __init__(self, news_dataset : pd.DataFrame, new_data: dict, **kwargs):
        self.news = news_dataset
        self.user_data = new_data
        self.num_rec = kwargs.get("num_rec", 3)

    def recommend(self,) -> pd.DataFrame:

        lable_encoders = {}
        for column in ["성별", "나이","직업", "거주지"]:
            lable_encoders[column] = LabelEncoder()
            self.user_data[column] = lable_encoders[column].fit_transform([self.user_data[column]])

        # 모델과 토크나이저 로드
        model_name = "klue/roberta-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        # 유저 데이터 인코딩
        def encode_user_data(user_data):
            label_encoders = {}
            for column in ["성별", "직업", "거주지"]:
                label_encoders[column] = LabelEncoder()
                label_encoders[column].fit(user_data[column])
                user_data[column] = label_encoders[column].transform(user_data[column])

            # Create a single string to represent the user data
            user_string = f"{self.user_data['성별'][0]} {self.user_data['나이'][0]} {self.user_data['직업'][0]} {self.user_data['거주지'][0]}"

            inputs = tokenizer(
                user_string,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            return inputs
        
        # 뉴스 데이터 인코딩
        def encoded_news_data(news_data):
            inputs = tokenizer(
                news_data["title"].tolist(),
                news_data["content"].tolist(),
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            return inputs
        
        user_inputs = encode_user_data(self.user_data)
        news_inputs = encoded_news_data(self.news)

        # PLM을 통한 추론
        with torch.no_grad():
            user_output = model(**user_inputs)
            news_output = model(**news_inputs)
        
        user_embeddings = user_output.pooler_output
        news_embeddings = news_output.pooler_output 
        similarity = cosine_similarity(user_embeddings, news_embeddings)

        # 추천결과
        num_recommendations = self.num_rec
        sorted_indices = (-similarity).argsort().squeeze()
        recommended_news = self.news.iloc[sorted_indices[:num_recommendations]]

        return recommended_news