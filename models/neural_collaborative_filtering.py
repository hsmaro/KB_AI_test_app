import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

import math
from torch import nn, optim
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def read_dataset(data_path):
    df = pd.read_csv(os.path.join(data_path, "hs_rating_v2.csv")) # 추후 다시 원상복구하기
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=29, shuffle=True)
    return train_df, val_df


class DatasetLoader:

    def __init__(self, data_path):

        self.train_df, val_temp_df = read_dataset(data_path)
        
        self.min_rating = min(self.train_df.rating)
        self.max_rating = self.train_df.rating.max()

        self.unique_user = self.train_df.user_id.unique()
        self.num_users = len(self.unique_user)
        self.user2idx = {ori : idx for idx, ori in enumerate(self.unique_user)}

        self.unique_news = self.train_df.label.unique()
        self.num_news = len(self.unique_news)
        self.news2idx = {ori : idx for idx, ori in enumerate(self.unique_news)}

        self.val_df = val_temp_df[val_temp_df.user_id.isin(self.unique_user) & val_temp_df.label.isin(self.unique_news)]
        
        self.user_info = self.train_df[['user_id', 'gender', 'age', 'occupation', 'address']].drop_duplicates()
        for _, row in self.user_info.iterrows():
            user_id, gender, age, occupation, address = row
            self.user2idx[user_id] = {'user_idx': self.user2idx[user_id],
                                      'gender': gender,
                                      'age': age,
                                      'occupation': occupation,
                                      'address': address}

    def generate_trainset(self):
        X_train = pd.DataFrame({"user": self.train_df.user_id.map(lambda x: self.user2idx[x]['user_idx']),
                                "news": self.train_df.label.map(self.news2idx),
                                "gender": self.train_df.user_id.map(lambda x: self.user2idx[x]['gender']),
                                "age": self.train_df.user_id.map(lambda x: self.user2idx[x]['age']),
                                "occupation": self.train_df.user_id.map(lambda x: self.user2idx[x]['occupation']),
                                "address": self.train_df.user_id.map(lambda x: self.user2idx[x]['address'])})
        y_train = self.train_df["rating"].astype(np.float32)

        return X_train, y_train

num_genders = 2
num_ages = 5
num_occupations = 10
num_addresses = 6

class FeedForwardEmbedNN(nn.Module):

    def __init__(self, n_users, n_news, hidden, dropouts, n_factors, embedding_dropout):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, n_factors)
        self.news_emb = nn.Embedding(n_news, n_factors)
        self.gender_emb = nn.Embedding(num_genders, n_factors)  
        self.age_emb = nn.Embedding(num_ages, n_factors)  
        self.occupation_emb = nn.Embedding(num_occupations, n_factors) 
        self.address_emb = nn.Embedding(num_addresses, n_factors)  
        
        self.drop = nn.Dropout(embedding_dropout)
        self.hidden_layers = nn.Sequential(*list(self.generate_layers(n_factors*6, hidden, dropouts))) 
        self.fc = nn.Linear(hidden[-1], 1)

    def generate_layers(self, n_factors, hidden, dropouts):
        assert len(dropouts) == len(hidden)

        idx = 0
        while idx < len(hidden):
            if idx == 0:
                yield nn.Linear(n_factors, hidden[idx])
            else:
                yield nn.Linear(hidden[idx-1], hidden[idx])
            yield nn.ReLU()
            yield nn.Dropout(dropouts[idx])

            idx += 1

    def forward(self, users, news, gender, age, occupation, address, min_rating=0.5, max_rating=5):
        user_embeds = self.user_emb(users)
        news_embeds = self.news_emb(news)
        gender_embeds = self.gender_emb(gender)
        age_embeds = self.age_emb(age)
        occupation_embeds = self.occupation_emb(occupation)
        address_embeds = self.address_emb(address)

        concat_features = torch.cat([user_embeds, news_embeds, gender_embeds, age_embeds, occupation_embeds, address_embeds], dim=1)
        x = F.relu(self.hidden_layers(concat_features))
        out = torch.sigmoid(self.fc(x))
        out = (out * (max_rating - min_rating)) + min_rating

        return out

    def predict(self, users, news, gender, age, occupation, address):
        output_scores = self.forward(users, news, gender, age, occupation, address)
        return output_scores


class BatchIterator:

    def __init__(self, X, y, batch_size=32, shuffle=True):
        X, y = np.asarray(X), np.asarray(y)

        if shuffle:
            index = np.random.permutation(X.shape[0])
            X, y = X[index], y[index]

        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_batches = int(math.ceil(X.shape[0] // batch_size))
        self._current = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self._current >= self.n_batches:
            raise StopIteration()
        k = self._current
        self._current += 1
        bs = self.batch_size
        return self.X[k * bs:(k + 1) * bs], self.y[k * bs:(k + 1) * bs]

def batches(X, y, bs=512, shuffle=True):
    for x_batch, y_batch in BatchIterator(X, y, bs, shuffle):
        x_batch = torch.LongTensor(x_batch)
        y_batch = torch.FloatTensor(y_batch)
        yield x_batch, y_batch.view(-1, 1)

def model_train(ds, config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    X_train, y_train = ds.generate_trainset()
    X_valid, y_valid = ds.generate_validset()
    print(f"TrainSet Info: {ds.num_users} users, {ds.num_news} news")

    model = FeedForwardEmbedNN(
        n_users=ds.num_users, n_news=ds.num_news,
        hidden=config["hidden_layers"], dropouts=config["dropouts"],
        n_factors=config["num_factors"], embedding_dropout=config["embedding_dropout"]
    )
    model.to(device)

    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    max_patience = config["total_patience"]
    num_patience = 0
    best_loss = np.inf

    criterion = nn.MSELoss(reduction="sum")
    criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    result = dict()
    for epoch in tqdm(range(num_epochs)):
        training_loss = 0.0
        for batch in batches(X_train, y_train, shuffle=True, bs=batch_size):
            x_batch, y_batch = [b.to(device) for b in batch]
            users = x_batch[:, 0]
            news = x_batch[:, 1]
            gender = x_batch[:, 2]  
            age = x_batch[:, 3]  
            occupation = x_batch[:, 4] 
            address = x_batch[:, 5] 

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(users, news, gender, age, occupation, address, ds.min_rating, ds.max_rating)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            training_loss += loss.item()
        result["train"] = training_loss / len(X_train)

        val_outputs = model(torch.LongTensor(X_valid.user.values).to(device),
                            torch.LongTensor(X_valid.news.values).to(device),
                            torch.LongTensor(X_valid.gender.values).to(device),
                            torch.LongTensor(X_valid.age.values).to(device),
                            torch.LongTensor(X_valid.occupation.values).to(device),
                            torch.LongTensor(X_valid.address.values).to(device),
                            ds.min_rating, ds.max_rating)
        val_loss = criterion(val_outputs.to(device), torch.FloatTensor(y_valid.values).view(-1, 1).to(device))
        result["val"] = float((val_loss / len(X_valid)).data)

        if val_loss < best_loss:
            print("Save new model on epoch: %d" % (epoch + 1))
            best_loss = val_loss
            result["best_loss"] = val_loss
            torch.save(model.state_dict(), config["save_path"])
            num_patience = 0
        else:
            num_patience += 1

        print(f'[epoch: {epoch+1}] train: {result["train"]} - val: {result["val"]}')

        if num_patience >= max_patience:
            print(f"Early Stopped after epoch {epoch+1}")
            break

    return result

def model_valid(user_id_list, news_id_list, data_path, config):
    dataset = DatasetLoader(data_path)
    processed_test_input_df = pd.DataFrame({
        "user_id": [dataset.user2idx[x]['user_idx'] for x in user_id_list],
        "label": [dataset.news2idx[x] for x in news_id_list],
        "gender": [dataset.user2idx[x]['gender'] for x in user_id_list],
        "age": [dataset.user2idx[x]['age'] for x in user_id_list],
        "occupation": [dataset.user2idx[x]['occupation'] for x in user_id_list],
        "address": [dataset.user2idx[x]['address'] for x in user_id_list]
    })

    # 학습한 모델 load하기 
    my_model = FeedForwardEmbedNN(dataset.num_users, dataset.num_news,
                       config["hidden_layers"], config["dropouts"], config["num_factors"], config["embedding_dropout"]).to("cpu")
    my_model.load_state_dict(torch.load("params.data", map_location=torch.device('cpu'))) #  - gpu로 돌릴 경우에는 제거
    prediction_outputs = my_model.predict(users=torch.LongTensor(processed_test_input_df.user_id.values),
                                        news=torch.LongTensor(processed_test_input_df.label.values),
                                        gender=torch.LongTensor(processed_test_input_df.gender.values),
                                        age=torch.LongTensor(processed_test_input_df.age.values),
                                        occupation=torch.LongTensor(processed_test_input_df.occupation.values),
                                        address=torch.LongTensor(processed_test_input_df.address.values))

    return prediction_outputs


config = {
  "num_factors": 16,
  "hidden_layers": [64, 32, 16],
  "embedding_dropout": 0.05,
  "dropouts": [0.3, 0.3, 0.3],
  "learning_rate": 1e-5,
  "weight_decay": 1e-5,
  "batch_size": 8,
  "num_epochs": 3,
  "total_patience": 30,
  "save_path": "params2.data"
}