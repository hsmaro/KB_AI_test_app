import pandas as pd
import numpy as np
import gensim
from gensim import corpora
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
from pprint import pprint


'''
LDA 기반의 추천 모델링
기사의 제목과 본문을 통하여 토픽을 추출하고 추출된 토픽을 가지고 콘텐츠 기반의 추천을 한다.
기존의 모델을 학습한 상태로 그 모델을 기반으로 하여 새로운 뉴스를 크롤링하여 실시간 추천에 용이
추론 속도가 가장 빠르기에 실시간에 적합
단순 기사의 제목과 본문만 있으면 사용가능하기 때문에 콜드 스타트 문제에 대응 가능
'''
np.random.seed(29)

class LDAContentRecModel():
    def __init__(self, user_dataset: pd.DataFrame, news_dataset: pd.DataFrame, new_data: dict, **kwargs):
        self.user = user_dataset
        self.news = news_dataset
        self.new_data = new_data
        self.topics = kwargs.get("topics", 6)
        self.n_epochs = kwargs.get("n_epochs", 30)

    def recommend(self) -> str:
        all_data = []
        all_data.extend([" ".join(map(str, row)) for _, row in self.user.iterrows()])        
        all_data.extend(self.news["title"])
        all_data.extend(self.news["content"])

        processed_data = [simple_preprocess(data) for data in all_data]

        dictionary = corpora.Dictionary(processed_data)
        corpus = [dictionary.doc2bow(text) for text in processed_data]

        lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=self.topics, passes=self.n_epochs)

        new_user_profile_text = self.new_data.get("profile_text", "")  
        new_user_profile_bow = dictionary.doc2bow(simple_preprocess(new_user_profile_text))
        user_topic_vector = lda_model[new_user_profile_bow]

        user_interest_topic = user_topic_vector[0][0]

        news_title = self.news["title"]
        news_content = self.news["content"]

        for i, doc_topics in enumerate(lda_model.get_document_topics(corpus)):
            sorted_topics = sorted(doc_topics, key=lambda x: x[1], reverse=True)
            top_topic = sorted_topics[0][0]

            if top_topic == user_interest_topic and i >= len(self.user) + len(news_title):
                pprint(f"제목 : {news_title[i - len(self.user) - len(news_title)]}")
                pprint(f"내용 : {news_content[i - len(self.user) - len(news_title)]}")
                return ("추천뉴스 기사입니다.")

        return "추천할 뉴스 기사가 없습니다."