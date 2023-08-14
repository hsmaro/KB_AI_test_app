## 사이드바에 기사 정렬
### 일렬로 정렬 완료 - 기사추천은 params의 수정이 필요
import streamlit as st
import os
import pandas as pd
from pprint import pprint
import textwrap
import random
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from utils.preprocessing import UserLabelProcessor
from models.neural_collaborative_filtering_ab_v2 import DatasetLoader, model_valid

#font_path = "./data/NanumGothic"  # 나눔고딕 폰트 경로
#fontprop = fm.FontProperties(fname=font_path, size=12)

# define path
data_path = os.path.join(os.getcwd(), "data")
#backup_path = os.path.join(data_path, "backup")
#os.makedirs(backup_path, exist_ok=True)
user_path = os.path.join(data_path, "user_db_v4.csv") #        "hs_user_db_v2.csv"

## pick user
def user_pick():
    user  = pd.read_csv(user_path) # user의 기본정보
    idx = random.randint(0, user.shape[0]-1)
    id = f"user_{idx+1}" # user_id 생성
    selected_user = user[user["user_id"]==id] # 해당 user의 행만 추출
    st.session_state.selected_user = selected_user
    
    # 해당 user의 정보 추출
    user_gender = selected_user.loc[idx, "gender"] 
    user_age  = selected_user.loc[idx, "age"]
    user_occupation = selected_user.loc[idx, "occupation"]
    user_address = selected_user.loc[idx, "address"]

    # st.session_state 페이지가 종료되어도 값이 남아있다.
    st.session_state.idx = idx
    st.session_state.user_id = id
    st.session_state.user_gender = user_gender
    st.session_state.user_age = user_age
    st.session_state.user_occupation = user_occupation
    st.session_state.user_address = user_address

## pick question
def qa():
    quiz = pd.read_csv("./data/e_ox_v3.csv") # 문제 데이터 # 문답 데이터 호출
    
    ## 문제 선택
    quiz_id = random.randint(0, quiz.shape[0]-1) # 0부터 해당 열의 갯수만큼 중 랜덤
        
    qa_quiz = quiz.loc[quiz_id, "question"] # 질문
    qa_ans = quiz.loc[quiz_id, "answer"] # 정답
    qa_label = quiz.loc[quiz_id, "label"] # 카테고리
    qa_context = quiz.loc[quiz_id, "info"] # 간단한 해설
        
    ## st.session_state 에 저장
    st.session_state.quiz_id = quiz_id # id 고정
    st.session_state.qa_quiz = qa_quiz
    st.session_state.qa_ans = qa_ans
    st.session_state.qa_label = qa_label
    st.session_state.qa_context = qa_context

    qa_txt = textwrap.fill(st.session_state.qa_quiz, width=35) # 최대 35자로 길이 제한
    st.session_state.qa_txt = qa_txt

## question_check
def check(qa_ans, user_ans):
    image_O = "./data/red_O_256.png"
    image_X = "./data/red_X_256.png"
    if qa_ans == 0:
        img = image_O
    else:
        img = image_X
    
    
    if qa_ans == user_ans: # user의 정답과 실제 정답 비교 후 문구생성
        ox = 0
        #text = "정답 입니다."
        st.success("정답 입니다.")
    else:
        ox = 1
        #text = "오답 입니다."
        st.warning("오답입니다.")
    st.session_state.img = img
    st.session_state.ox = ox
    #st.session_state.text = text
    
## news_rec - 기사추천
def news_rec(input_csv_path, user_csv_path, output_csv_path, id):
    # Preprocess user information and save to a CSV file
    news = pd.read_csv("./data/news_db_v2.csv") # news 데이터
    processor = UserLabelProcessor(input_csv_path, user_csv_path, output_csv_path)
    processor.melt_and_save()
    
    data_path = "./data/" 

    config = {
        "num_factors": 8,
        "hidden_layers": [512, 256, 128],
        "embedding_dropout": 0.05,
        "dropouts": [0.3, 0.3, 0.3],
        "learning_rate": 1e-3,
        "weight_decay": 1e-3,
        "batch_size": 2048,
        "num_epochs": 300,
        "total_patience": 30,
        "save_path": "params2.data"
        }

    
    news_id_list = ["금융", "증권", "부동산", "글로벌경제", "생활경제", "경제일반"]
    user_id = id
    user_id_list = [user_id] * len(news_id_list)

    pred_results = model_valid(user_id_list, news_id_list, data_path, config)

    result_df = pd.DataFrame({
        "userId": user_id_list,
        "label": news_id_list,
        "pred_ratings": [float(r.detach().numpy()) for r in pred_results],
    })

    rec_label = result_df.sort_values(by=["pred_ratings"], ascending=[False]).iloc[-1, 1]
    st.session_state.rec_label = rec_label

    # 뉴스 추천
    rec_news_zip = news[news["label"]==rec_label].reset_index(drop=True) # news의 라벨이 rec_label인 것들만 특정 후 인덱스 초기화
    
    #무작위 5개의 기사 추출 
    news_idx = random.sample(rec_news_zip.index.tolist(), 5) 
    
    ## 추천된 뉴스 title과 content 구분
    news_titles = rec_news_zip.loc[news_idx, "title"].to_list() # 뉴스 제목 리스트 반환 news_idx
    news_contents = rec_news_zip.loc[news_idx, "content"].to_list() # 뉴스 본문 리스트 반환 news_idx
    
    return rec_label, news_titles, news_contents

# user_update
def update_user_db(label, ox, idx):
    user_db  = pd.read_csv(user_path) # user의 기본정보
    
    if ox == 0: # 정답일 경우
        user_db.loc[idx, "total"] += 1
        user_db.loc[idx, f"{label}_tot"] += 1
        user_db.loc[idx, f"{label}_ans"] += 1
    else:
        user_db.loc[idx, "total"] += 1
        user_db.loc[idx, f"{label}_tot"] += 1
    
    selected_user = user_db[user_db["user_id"]==st.session_state.user_id] # 해당 user의 행만 추출
    st.session_state.selected_user = selected_user
    
    split_user_db = user_db.iloc[:, 0:11]
    
    user_db.to_csv(user_path, encoding="utf-8", index=False) ## 나중에 바꾸기
    split_user_db.to_csv(os.path.join(data_path, "split_db_v4.csv"), encoding="utf-8", index=False) #   "hs_split_db_v2.csv"
    
    return user_db
'''        
def see_graph(user_db):
     u_df = user_db.iloc[st.session_state.idx:st.session_state.idx+1, 5:11].T.reset_index() # 5:11 -> 금융, 증권 ,부동산, 글로벌경제, 생활경제, 일반경제
     u_df.columns=["label", "정답률"]
     fig = px.bar(u_df, x="label", y="정답률", title=f"{st.session_state.user_id}의 정답률",
                  hover_name = "label",
                  hover_data={"정답률": ":.2f", "label":False})
     fig.update_xaxes(title='') # x축 이름 제거
     fig.update_yaxes(title='') # y축 이름 제거
     fig.update_yaxes(showgrid=False) # y축 그리드(눈금) 제거
     st.plotly_chart(fig) # 차트랑 문제가 같이 뜨게 적용

def see_matplotlib(user_db):
    u_df = user_db.iloc[st.session_state.idx:st.session_state.idx+1, 5:11].T.reset_index()
    u_df.columns=["label", "정답률"]
    # Figure와 Axes 객체 생성
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(x=u_df["label"], height=u_df["정답률"])
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}%", ha="center", va="bottom", fontproperties=fontprop)
    
    st.pyplot(fig)
'''

def rec_list(news_title, news_content):
    
    ## 기사 출력
    #st.title("News")
    st.write(f"### {news_title}")
    news_txt = textwrap.fill(news_content, width=30)
    st.text(news_txt)

## show_predict_page
def show_predict_page():
    st.set_page_config(initial_sidebar_state='expanded')
    #st.set_page_config(initial_sidebar_state="auto")
    if not hasattr(st.session_state, 'user_id'):
        user_pick() # 일단 user_id만 사용
    
    if not hasattr(st.session_state, 'quiz_id'):
        qa() 
        
    if not hasattr(st.session_state, 'rec_label'):
        st.title("Quiz")
        st.write(st.session_state.qa_txt)
        user_ans = st.radio("정답", ("O", "X"))
        if user_ans == "O":
            user_ans = 0
        else:
            user_ans = 1
     
        qa_submit = st.button("submit", key="qa_button")
        
        if qa_submit:
            st.session_state.user_ans = user_ans
            ## 정답 여부 판별
            check(st.session_state.qa_ans, st.session_state.user_ans)
            #st.write(st.session_state.text)
            st.write("### 정답: ")
            st.image(st.session_state.img)
            st.write("#### 해설")
            st.write(st.session_state.qa_context)
            #col1, col2 = st.columns(2)  # 화면을 두 개의 열로 분할
            #with col1:
            #    st.image(st.session_state.img, width=200)  # 이미지 표시

            #with col2:
            #    st.write("#### 해설")
            #    st.write(st.session_state.qa_context)  # 설명 표시 
        
    if hasattr(st.session_state, 'user_ans') and not hasattr(st.session_state, 'rec_label'):
        ## user 업데이트
        user_db = update_user_db(st.session_state.qa_label, st.session_state.ox, st.session_state.idx)
        
        #see_graph(user_db)
        #see_matplotlib(user_db)
        with st.spinner('뉴스를 추천 중입니다...'):
            rec_label, news_titles, news_contents = news_rec(os.path.join(data_path, "split_db_v4.csv"), #        "hs_split_db_v2.csv"
                                                                os.path.join(data_path, "split_db_v4.csv"), #        "hs_split_db_v2.csv"
                                                                os.path.join(data_path, "rating_v4.csv"), #       "hs_rating_v2.csv"
                                                                st.session_state.user_id)
            st.session_state.news_titles = news_titles
            st.session_state.news_contents = news_contents
    
    if hasattr(st.session_state, 'rec_label') and hasattr(st.session_state, 'user_ans'):
    ## title 5개 출력
        st.sidebar.title(f"추천 카테고리: {st.session_state.rec_label}")
        for idx, news_title in enumerate(st.session_state.news_titles):
            unique_key = f"sidebar_button_{idx}"
            if st.sidebar.button(news_title, key=unique_key):
                st.session_state.news_title = news_title
                st.session_state.news_content = st.session_state.news_contents[idx]
                rec_list(st.session_state.news_title, st.session_state.news_content)

