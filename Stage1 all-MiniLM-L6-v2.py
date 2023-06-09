# ====================================================================================================

!pip install kaggle
from google.colab import drive

drive.mount('/content/drive')
import os
import json
f = open("/content/drive/MyDrive/kaggle.json", 'r')
json_data = json.load(f) 
os.environ['KAGGLE_USERNAME'] = json_data['username']
os.environ['KAGGLE_KEY'] = json_data['key']
!kaggle competitions download -c learning-equality-curriculum-recommendations
!unzip '/content/learning-equality-curriculum-recommendations.zip'

!pip -qqq install sentence-transformers
!pip install datasets
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from sentence_transformers import SentenceTransformer, models, InputExample, losses
from datasets import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold


# ====================================================================================================

DATA_PATH = "/content/"
topics = pd.read_csv(DATA_PATH + "topics.csv")
content = pd.read_csv(DATA_PATH + "content.csv")
correlations = pd.read_csv(DATA_PATH + "correlations.csv")

seq_len = 64

# 以下、breadcrumbsを作るためのコード
topics_bread = pd.read_csv(DATA_PATH + "topics.csv", index_col=0).fillna({"title": "", "description": ""})

# トピッククラス
class Topic:
    def __init__(self, topic_id):
        self.id = topic_id
        
    @property
    # 親を求める関数
    def parent(self):
        parent_id = topics_bread.loc[self.id].parent
        if pd.isna(parent_id):   # 欠損値を確認。親IDがなければ何も返さない
            return None
        else:
            return Topic(parent_id)

    @property
    # 祖先(親を辿る＝トピックツリー)を求める関数
    def ancestors(self):
        ancestors = []
        parent = self.parent
        while parent is not None:   # 
            ancestors.append(parent)
            parent = parent.parent
        return ancestors
    
    # パンくずリスト(トピックツリーの子孫・祖先)を求める関数
    def get_breadcrumbs(self, separator=" >> ", include_self=True, include_root=True):  # 間のトークンの種類を変えてみる
        ancestors = self.ancestors
        if include_self:
            ancestors = [self] + ancestors
        if not include_root:
            ancestors = ancestors[:-1]
        return separator.join(reversed([a.title for a in ancestors]))
    
    # 拡張比較メソッド(eq,equal:==)の実装
    def __eq__(self, other):
        if not isinstance(other, Topic):   # otherがTopicクラスでなければ、falseを返す
            return False
        return self.id == other.id   # ==比較演算子を適用する

    # 名前インスタンスの定義外の呼び出しへの関数
    def __getattr__(self, name):
        return topics_bread.loc[self.id][name]   # nameが定義されていれば、普通に返す。定義外なら、AttributeErrorを返す

    # 文字列表現の特殊メソッド
    def __str__(self):
        return self.title
    
    # 引数に''を付けて返す特殊メソッド
    def __repr__(self):
        return f"<Topic(id={self.id}, title=\"{self.title}\")>"  # トピッククラスの中のトピックIDとタイトルを返す

# トピックツリー(breadcrumbs)のカラムを作成
topics['breadcrumbs'] = [Topic(topics_bread.index[i]).get_breadcrumbs() for i in range(len(topics))]  # やらかしたかも

del topics_bread


# ====================================================================================================

# only topics has_content
#topics = topics[topics.has_content == True]

# Fillna title
topics['title'].fillna("", inplace = True)
content['title'].fillna("", inplace = True)
# Fillna descriptions
topics['description'].fillna("", inplace = True)
content['description'].fillna("", inplace = True)
# Fillna text, breadcrumbs
content['text'].fillna("", inplace = True)
topics['breadcrumbs'].fillna("", inplace = True)

# Step1: Retriverより、CV(fold=4,5,10)を採用
def cv_split(train, n_folds, seed):
    kfold = KFold(n_splits = n_folds, shuffle = True, random_state = seed)
    for num, (train_index, val_index) in enumerate(kfold.split(train)):
        train.loc[val_index, 'fold'] = int(num)
    train['fold'] = train['fold'].astype(int)
    return train

kfolds = cv_split(correlations, 4, 42)
# foldしたcorr.dfを使うため、fold分割済みのcorr.dfを別途.csvファイル形式で保存する必要がある
corre = kfolds
corre.to_csv('kfold_correlations.csv', index=False)

correlations = kfolds[kfolds.fold != 0]  # !=

topics.rename(columns=lambda x: "topic_" + x, inplace=True)
content.rename(columns=lambda x: "content_" + x, inplace=True)

correlations["content_id"] = correlations["content_ids"].str.split(" ")
corr = correlations.explode("content_id").drop(columns=["content_ids"])

corr = corr.merge(topics, how="left", on="topic_id")
corr = corr.merge(content, how="left", on="content_id")

corr['topics'] = corr['topic_title'] + '[SEP]' + corr['topic_description'] + '[SEP]' + corr['topic_breadcrumbs'] 
corr['contents'] = corr['content_title'] + '[SEP]' + corr['content_description'] + '[SEP]' + corr['content_text']
corr['topics'] = [corr['topics'][i][:seq_len] for i in range(len(corr))]
corr['contents'] = [corr['contents'][i][:seq_len] for i in range(len(corr))]

corr["set"] = corr[["topics", "contents"]].values.tolist()

train_df = pd.DataFrame(corr["set"])

dataset = Dataset.from_pandas(train_df)

train_examples = []
train_data = dataset["set"]
n_examples = dataset.num_rows

for i in range(n_examples):
    example = train_data[i]
    if example[0] == None: #remove None
        print(example)
        continue        
    # setにあるやつを全部一つの文章(リスト)にしてぶち込んでいく
    train_examples.append(InputExample(texts=[str(example[0]), str(example[1])]))


# ====================================================================================================

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=288)
train_loss = losses.MultipleNegativesRankingLoss(model=model)
num_epochs = 30
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1) #10% of train data

model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          save_best_model = True,
          #steps_per_epoch = 500,
          output_path='/content/drive/My Drive/Colab Notebooks/all-MiniLM-L6-v2-exp_batch288_fold4_stage1_epochs30',  # ベストモデルを直接保存したい
          use_amp = True,
          warmup_steps=warmup_steps)
