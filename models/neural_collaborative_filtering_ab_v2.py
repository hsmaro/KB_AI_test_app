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
    df = pd.read_csv(os.path.join(data_path, "train.csv")) # 
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=29, shuffle=True)
    return train_df, val_df

def read_testdataset(data_path):
    df = pd.read_csv(os.path.join(data_path, "ab_with.csv")) # 
    return df


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
        
        self.user_info = self.train_df[['user_id', 'gender', 'age', 'occupation', 'address', 'acc_avg', 'click_probs']].drop_duplicates()
        for _, row in self.user_info.iterrows():
            user_id, gender, age, occupation, address, acc_avg, click_probs = row
            self.user2idx[user_id] = {'user_idx': self.user2idx[user_id],
                                      'gender': gender,
                                      'age': age,
                                      'occupation': occupation,
                                      'address': address,
                                      'acc_avg': acc_avg,
                                      'click_probs': click_probs}

    def generate_trainset(self):
        X_train = pd.DataFrame({"news": self.train_df.label.map(self.news2idx),
                                "gender": self.train_df.user_id.map(lambda x: self.user2idx[x]['gender']),
                                "age": self.train_df.user_id.map(lambda x: self.user2idx[x]['age']),
                                "occupation": self.train_df.user_id.map(lambda x: self.user2idx[x]['occupation']),
                                "address": self.train_df.user_id.map(lambda x: self.user2idx[x]['address']),
                                "acc_avg": self.train_df.user_id.map(lambda x: self.user2idx[x]['acc_avg']),
                                "click_probs": self.train_df.user_id.map(lambda x: self.user2idx[x]['click_probs'])})
        y_train = self.train_df["rating"].astype(np.float32)

        return X_train, y_train

    def generate_validset(self):
        X_valid = pd.DataFrame({"news": self.val_df.label.map(self.news2idx),
                                "gender": self.val_df.user_id.map(lambda x: self.user2idx[x]['gender']),
                                "age": self.val_df.user_id.map(lambda x: self.user2idx[x]['age']),
                                "occupation": self.val_df.user_id.map(lambda x: self.user2idx[x]['occupation']),
                                "address": self.val_df.user_id.map(lambda x: self.user2idx[x]['address']),
                                "acc_avg": self.val_df.user_id.map(lambda x: self.user2idx[x]['acc_avg']),
                                "click_probs": self.val_df.user_id.map(lambda x: self.user2idx[x]['click_probs'])})
        y_valid = self.val_df["rating"].astype(np.float32)

        return X_valid, y_valid
    

class TestDatasetLoader:

    def __init__(self, data_path):

        self.test_df = read_testdataset(data_path)
        
        self.min_rating = min(self.test_df.rating)
        self.max_rating = self.test_df.rating.max()

        self.unique_user = self.test_df.user_id.unique()
        self.num_users = len(self.unique_user)
        self.user2idx = {ori : idx for idx, ori in enumerate(self.unique_user)}

        self.unique_news = self.test_df.label.unique()
        self.num_news = len(self.unique_news)
        self.news2idx = {ori : idx for idx, ori in enumerate(self.unique_news)}
        
        self.user_info = self.test_df[['user_id', 'gender', 'age', 'occupation', 'address']].drop_duplicates()
        for _, row in self.user_info.iterrows():
            user_id, gender, age, occupation, address = row
            self.user2idx[user_id] = {'user_idx': self.user2idx[user_id],
                                      'gender': gender,
                                      'age': age,
                                      'occupation': occupation,
                                      'address': address}

    def generate_testset(self):
        X_test = pd.DataFrame({"user": self.test_df.user_id.map(lambda x: self.user2idx[x]['user_idx']),
                                "news": self.test_df.label.map(self.news2idx),
                                "gender": self.test_df.user_id.map(lambda x: self.user2idx[x]['gender']),
                                "age": self.test_df.user_id.map(lambda x: self.user2idx[x]['age']),
                                "occupation": self.test_df.user_id.map(lambda x: self.user2idx[x]['occupation']),
                                "address": self.test_df.user_id.map(lambda x: self.user2idx[x]['address'])})
        y_test = self.test_df["rating"].astype(np.float32)

        return X_test, y_test

num_genders = 2
num_ages = 5
num_occupations = 12
num_addresses = 6

class FeedForwardEmbedNN(nn.Module):

    def __init__(self, n_news, hidden, dropouts, n_factors, embedding_dropout):
        super().__init__()
        self.news_emb = nn.Embedding(n_news, n_factors)
        self.gender_emb = nn.Embedding(num_genders, n_factors)  
        self.age_emb = nn.Embedding(num_ages, n_factors)  
        self.occupation_emb = nn.Embedding(num_occupations, n_factors) 
        self.address_emb = nn.Embedding(num_addresses, n_factors)  
        
        self.drop = nn.Dropout(embedding_dropout)
        self.hidden_layers = nn.Sequential(*list(self.generate_layers(n_factors*5 + 2, hidden, dropouts))) 
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

    def forward(self, news, gender, age, occupation, address, acc_avg, click_probs, min_rating=0., max_rating=1.):
        news_embeds = self.news_emb(news)
        gender_embeds = self.gender_emb(gender)
        age_embeds = self.age_emb(age)
        occupation_embeds = self.occupation_emb(occupation)
        address_embeds = self.address_emb(address)

        acc_avg = torch.tensor(acc_avg)
        acc_avg = acc_avg.unsqueeze(1)

        click_probs = torch.tensor(click_probs)
        click_probs = click_probs.unsqueeze(1)


        concat_features = torch.cat([news_embeds, gender_embeds, age_embeds, occupation_embeds, address_embeds, acc_avg, click_probs], dim=1)
        x = F.relu(self.hidden_layers(concat_features))
        out = torch.sigmoid(self.fc(x))
        out = (out * (max_rating - min_rating)) + min_rating

        return out

    def predict(self, news, gender, age, occupation, address, acc_avg, click_probs):
        output_scores = self.forward(news, gender, age, occupation, address, acc_avg, click_probs)
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

        news = x_batch[:, 0]
        gender = x_batch[:, 1]
        age = x_batch[:, 2]
        occupation = x_batch[:, 3]
        address = x_batch[:, 4]
        acc_avg = x_batch[:, 5]
        click_probs = x_batch[:, 6]

        news = torch.LongTensor(news)
        gender = torch.LongTensor(gender)
        age = torch.LongTensor(age)
        occupation = torch.LongTensor(occupation)
        address = torch.LongTensor(address)
        acc_avg = torch.FloatTensor(acc_avg)
        click_probs = torch.FloatTensor(click_probs)

        y_batch = torch.FloatTensor(y_batch)
        yield (news, gender, age, occupation, address, acc_avg, click_probs), y_batch.view(-1, 1)


def model_train(ds, config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    X_train, y_train = ds.generate_trainset()
    X_valid, y_valid = ds.generate_validset()
    print(f"TrainSet Info: {ds.num_users} users, {ds.num_news} news")

    model = FeedForwardEmbedNN(n_news=6,
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
            (news, gender, age, occupation, address, acc_avg, click_probs), y_batch = batch

            news = news.to(device)
            gender = gender.to(device)
            age = age.to(device)
            occupation = occupation.to(device)
            address = address.to(device)
            acc_avg = acc_avg.to(device)
            click_probs = click_probs.to(device)

            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(news, gender, age, occupation, address, acc_avg, click_probs, ds.min_rating, ds.max_rating)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            training_loss += loss.item()
        result["train"] = training_loss / len(X_train)

        val_outputs = model(torch.LongTensor(X_valid.news.values).to(device),
                            torch.LongTensor(X_valid.gender.values).to(device),
                            torch.LongTensor(X_valid.age.values).to(device),
                            torch.LongTensor(X_valid.occupation.values).to(device),
                            torch.LongTensor(X_valid.address.values).to(device),
                            torch.FloatTensor(X_valid.acc_avg.values).to(device),
                            torch.FloatTensor(X_valid.click_probs.values).to(device),
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
    # test_dataset = TestDatasetLoader(data_path)
    dataset = DatasetLoader(data_path)
    processed_test_input_df = pd.DataFrame({
        "label": [dataset.news2idx[x] for x in news_id_list],
        "gender": [dataset.user2idx[x]['gender'] for x in user_id_list],
        "age": [dataset.user2idx[x]['age'] for x in user_id_list],
        "occupation": [dataset.user2idx[x]['occupation'] for x in user_id_list],
        "address": [dataset.user2idx[x]['address'] for x in user_id_list],
        "acc_avg": [dataset.user2idx[x]['acc_avg'] for x in user_id_list],
        "click_probs": [dataset.user2idx[x]['click_probs'] for x in user_id_list]
    })

    # 학습한 모델 load하기 
    my_model = FeedForwardEmbedNN(dataset.num_news,
                       config["hidden_layers"], config["dropouts"], config["num_factors"], config["embedding_dropout"])
    my_model.load_state_dict(torch.load("params2.data"))
    prediction_outputs = my_model.predict(news=torch.LongTensor(processed_test_input_df.label.values),
                                        gender=torch.LongTensor(processed_test_input_df.gender.values),
                                        age=torch.LongTensor(processed_test_input_df.age.values),
                                        occupation=torch.LongTensor(processed_test_input_df.occupation.values),
                                        address=torch.LongTensor(processed_test_input_df.address.values),
                                        acc_avg=torch.FloatTensor(processed_test_input_df.acc_avg.values),
                                        click_probs=torch.FloatTensor(processed_test_input_df.click_probs.values))

    return prediction_outputs




