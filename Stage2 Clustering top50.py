# =========================================================================================
# Libraries
# =========================================================================================
import os
import gc
import time
import math
import random
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint
import tokenizers
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_cosine_schedule_with_warmup, DataCollatorWithPadding
import cupy as cp
from cuml.metrics import pairwise_distances
from cuml.neighbors import NearestNeighbors
from sklearn.model_selection import KFold
from pathlib import Path
%env TOKENIZERS_PARALLELISM=false
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================================================================================
# Configurations
# =========================================================================================
class CFG:
    num_workers = 4
    #model = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    model = "/kaggle/input/finetuning-model-11/all-MiniLM-L6-v2-exp_batch288_fold4_bread_epochs30"
    #model = "/kaggle/input/k/minar44/multiple-negatives-ranking-loss-lecr/all-MiniLM-L6-v2-MNR-tuned-fold0/model.pth"
    tokenizer = AutoTokenizer.from_pretrained(model)
    batch_size = 256  # batch256, 512 で変わるのか検証 → 大差ない
    top_n = 50
    seed = 42
    
# =========================================================================================
# Data Loading
# =========================================================================================
def print_markdown(md):
    display(Markdown(md))
    
def read_data(cfg):
    class Topic:
        def __init__(self, topic_id):
            self.id = topic_id

        @property
        def parent(self):
            parent_id = topics_df.loc[self.id].parent  # topics_df → topics(?)
            if pd.isna(parent_id):
                return None
            else:
                return Topic(parent_id)

        @property
        def ancestors(self):
            ancestors = []
            parent = self.parent
            while parent is not None:
                ancestors.append(parent)
                parent = parent.parent
            return ancestors
        
        @property
        def siblings(self):
            if not self.parent:
                return []
            else:
                return [topic for topic in self.parent.children if topic != self]

        @property
        def content(self):
            if self.id in correlations_df.index:
                return [ContentItem(content_id) for content_id in correlations_df.loc[self.id].content_ids.split()]
            else:
                return tuple([]) if self.has_content else []

        def get_breadcrumbs(self, separator=" >> ", include_self=True, include_root=True):
            ancestors = self.ancestors
            if include_self:
                ancestors = [self] + ancestors
            if not include_root:
                ancestors = ancestors[:-1]
            return separator.join(reversed([a.title for a in ancestors]))
        
        @property
        def children(self):
            return [Topic(child_id) for child_id in topics_df[topics_df.parent == self.id].index]  # topics_df[topics_df.parent

        def subtree_markdown(self, depth=0):
            markdown = "  " * depth + "- " + self.title + "\n"
            for child in self.children:
                markdown += child.subtree_markdown(depth=depth + 1)
            for content in self.content:
                markdown += ("  " * (depth + 1) + "- " + "[" + content.kind.title() + "] " + content.title) + "\n"
            return markdown

        def __eq__(self, other):
            if not isinstance(other, Topic):
                return False
            return self.id == other.id

        def __getattr__(self, name):
            return topics_df.loc[self.id][name]  # topics_df.loc

        def __str__(self):
            return self.title

        def __repr__(self):
            return f"<Topic(id={self.id}, title=\"{self.title}\")>"

    class ContentItem:
        def __init__(self, content_id):
            self.id = content_id

        @property
        def topics(self):
            return [Topic(topic_id) for topic_id in topics_df.loc[correlations_df[correlations_df.content_ids.str.contains(self.id)].index].index]

        def __getattr__(self, name):
            return content_df.loc[self.id][name]

        def __str__(self):
            return self.title

        def __repr__(self):
            return f"<ContentItem(id={self.id}, title=\"{self.title}\")>"

        def __eq__(self, other):
            if not isinstance(other, ContentItem):
                return False
            return self.id == other.id
        
        def get_all_breadcrumbs(self, separator=" >> ", include_root=True):
            breadcrumbs = []
            for topic in self.topics:
                new_breadcrumb = topic.get_breadcrumbs(separator=separator, include_root=include_root)
                if new_breadcrumb:
                    new_breadcrumb = new_breadcrumb + separator + self.title
                else:
                    new_breadcrumb = self.title
                breadcrumbs.append(new_breadcrumb)
            return breadcrumbs
        
    data_dir = Path('/kaggle/input/learning-equality-curriculum-recommendations')
    topics_df = pd.read_csv(data_dir / "topics.csv").fillna({"title": "", "description": ""})
    content_df = pd.read_csv(data_dir / "content.csv", index_col=0).fillna("")
    correlations_df = pd.read_csv(data_dir / "correlations.csv", index_col=0)
    sample_submission = pd.read_csv(data_dir / 'sample_submission.csv')
   
    topics_df = topics_df.set_index('id')
    
    def cv_split(train, n_folds, seed):
        kfold = KFold(n_splits = n_folds, shuffle = True, random_state = seed)
        for num, (train_index, val_index) in enumerate(kfold.split(train)):
            train.loc[val_index, 'fold'] = int(num)
        train['fold'] = train['fold'].astype(int)
        return train

    correlations = pd.read_csv(data_dir / "correlations.csv")
    full_correlations = cv_split(correlations, 4, 42)
    correlations = full_correlations[full_correlations.fold == 0]

    topic_id_texts = []
    content_id_texts = []
    for topic_idx in tqdm(topics_df.index):
        tmp_topic = Topic(topic_idx)
        children = tmp_topic.children
        child = "" if len(children)==0 else children[0].description

        parent = tmp_topic.parent
        par = "" if parent is None else parent.description
        topic_repre = f"{tmp_topic.title}[SEP]{tmp_topic.description}[SEP]{tmp_topic.get_breadcrumbs()}"[:128] 

        topic_id_texts.append((topic_idx, topic_repre))

    for content_idx in tqdm(content_df.index):
        ct = ContentItem(content_idx)
        content_repre = f"{ct.title}[SEP]{ct.description}[SEP]{ct.text}"[:128]
        content_id_texts.append((content_idx, content_repre))

    topics = pd.DataFrame(data={'id':[item[0] for item in topic_id_texts], 
                             'set':[item[1] for item in topic_id_texts]})  # 'title'
    content = pd.DataFrame(data={'id':[item[0] for item in content_id_texts], 
                                 'set':[item[1] for item in content_id_texts]})  # 'title'
    
    del topics_df, content_df, correlations_df, sample_submission, topic_id_texts, content_id_texts
    gc.collect()

    print(' ')
    print('-' * 50)
    print(f"topics.shape: {topics.shape}")
    print(f"content.shape: {content.shape}")

    
    return topics, content, correlations, full_correlations

# =========================================================================================
# Prepare input, tokenize
# =========================================================================================
def prepare_input(text, cfg):
    inputs = cfg.tokenizer.encode_plus(
        text,  # 複数入力してもトークナイザーとして機能するか確認する
        #is_split_into_words=True,  # textをリスト形式で入力した時に必要らしい
        max_length = 128,
        truncation=True,
        return_tensors = None, 
        add_special_tokens = True, 
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype = torch.long)
    return inputs

# =========================================================================================
# Unsupervised dataset
# =========================================================================================
class uns_dataset(Dataset):
    def __init__(self, df, cfg):
        self.cfg = cfg
        #self.texts = df['title'].values 
        self.texts = df['set'].values  # ここで、title以外のdescriptionなどの入力を考える
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, item):
        inputs = prepare_input(self.texts[item], self.cfg)
        return inputs
    
# =========================================================================================
# Mean pooling class
# =========================================================================================
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

# =========================================================================================
# Unsupervised model
# =========================================================================================
class uns_model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.config = AutoConfig.from_pretrained(cfg.model)
        self.model = AutoModel.from_pretrained(cfg.model, config = self.config)
        self.pool = MeanPooling()
    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        feature = self.pool(last_hidden_state, inputs['attention_mask'])
        return feature
    def forward(self, inputs):
        feature = self.feature(inputs)
        return feature
    
# =========================================================================================
# Get embeddings
# =========================================================================================
def get_embeddings(loader, model, device):
    model.eval()
    preds = []
    for step, inputs in enumerate(tqdm(loader)):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds.to('cpu').numpy())
    preds = np.concatenate(preds)
    return preds

# =========================================================================================
# Get the amount of positive classes based on the total
# =========================================================================================
def get_pos_score(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    # 正答率(正解数 / 正解総数)
    int_true = np.array([len(x[0] & x[1]) / len(x[0]) for x in zip(y_true, y_pred)]) 
    return round(np.mean(int_true), 5)

# =========================================================================================
# F2 Score, Recall
def f2_score(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    tp = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    fp = np.array([len(x[1] - x[0]) for x in zip(y_true, y_pred)])
    fn = np.array([len(x[0] - x[1]) for x in zip(y_true, y_pred)])
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f2 = tp / (tp + 0.2 * fp + 0.8 * fn)
    return round(f2.mean(), 4), round(recall.mean(), 4)

# =========================================================================================
# Build our training set
# =========================================================================================
def build_training_set(topics, content, cfg):
    # Create lists for training
    topics_ids = []
    content_ids = []
    topics_set = []
    content_set = []
    targets = []
    folds = []
    # Iterate over each topic
    for k in tqdm(range(len(topics))):
        row = topics.iloc[k]
        topics_id = row['id']
        predictions = row['predictions'].split(' ')
        ground_truth = row['content_ids'].split(' ')
        topics_feature = row['set']  # set = title + description + breadcrumbs
        fold = row['fold']
        for pred in predictions:
            content_feature = content.loc[pred, 'set']
            topics_ids.append(topics_id)
            content_ids.append(pred)
            topics_set.append(topics_feature[:128])  # 文章(set)の文字数制限(len256) もしCVスコアが低いとしたら、ここが原因かも
            content_set.append(content_feature[:128])
            folds.append(fold)
            # If pred is in ground truth, 1 else 0
            # targetとは、予測が正解しているかどうかだった！
            if pred in ground_truth:
                targets.append(1)  # target=1: 正解
            else:
                targets.append(0)  # target=0: 不正解
    # Build training dataset
    train = pd.DataFrame(
        {'topics_ids': topics_ids, 
         'content_ids': content_ids, 
         'topics_set': topics_set,
         'content_set': content_set,
         'target': targets,
         'fold' : folds
        }
    )
    # Release memory
    del topics_ids, content_ids, topics_set, content_set, targets, folds
    gc.collect()
    return train
    
# =========================================================================================
# Get neighbors
# =========================================================================================
def get_neighbors(topics, content, cfg):
    # Create topics dataset
    topics_dataset = uns_dataset(topics, cfg)
    # Create content dataset
    content_dataset = uns_dataset(content, cfg)
    # Create topics and content dataloaders
    topics_loader = DataLoader(
        topics_dataset, 
        batch_size = cfg.batch_size, 
        shuffle = False, 
        collate_fn = DataCollatorWithPadding(tokenizer = cfg.tokenizer, padding = 'longest'),
        num_workers = cfg.num_workers, 
        pin_memory = True, 
        drop_last = False
    )
    content_loader = DataLoader(
        content_dataset, 
        batch_size = cfg.batch_size, 
        shuffle = False, 
        collate_fn = DataCollatorWithPadding(tokenizer = cfg.tokenizer, padding = 'longest'),
        num_workers = cfg.num_workers, 
        pin_memory = True, 
        drop_last = False
        )
    # Create unsupervised model to extract embeddings
    model = uns_model(cfg)
    model.to(device)
    # Predict topics
    topics_preds = get_embeddings(topics_loader, model, device)  # ここでembeddingを計算して、
    content_preds = get_embeddings(content_loader, model, device)  # title以外のsetを加えるということは、embeddingの計算が変わるということで、
    # ひいては、KNNを使って算出されるtop50のコンテンツの内容も変化する(recall,f2スコアの変化)
    # titleのみでの学習モデルで、setのベクトル計算だと、スコアが下がることがわかった(?)
    # では、全部盛りのsetで学習したモデルは、setのベクトル、titleのみのベクトル計算で最終的なスコアに変化は出るのか
    
    # Transfer predictions to gpu
    topics_preds_gpu = cp.array(topics_preds)
    content_preds_gpu = cp.array(content_preds)
    # Release memory
    torch.cuda.empty_cache()
    del topics_dataset, content_dataset, topics_loader, content_loader, topics_preds, content_preds
    gc.collect()
    # KNN model
    print(' ')
    print('Training KNN model...')
    neighbors_model = NearestNeighbors(n_neighbors = cfg.top_n, metric = 'cosine')
    neighbors_model.fit(content_preds_gpu)
    indices = neighbors_model.kneighbors(topics_preds_gpu, return_distance = False)
    predictions = []
    for k in tqdm(range(len(indices))):
        pred = indices[k]
        p = ' '.join([content.loc[ind, 'id'] for ind in pred.get()])
        predictions.append(p)
    topics['predictions'] = predictions
    # Release memory
    del topics_preds_gpu, content_preds_gpu, neighbors_model, predictions, indices, model
    gc.collect()
    return topics, content
     
# Read data
topics, content, correlations, full_correlations = read_data(CFG)

# Run nearest neighbors
topics, content = get_neighbors(topics, content, CFG)

# Merge with target and compute max positive score
topics_test = topics.merge(correlations, how = 'inner', left_on = ['id'], right_on = ['topic_id'])
pos_score = get_pos_score(topics_test['content_ids'], topics_test['predictions'])  # topicsかtopics_testか選んで！
f2_score, recall = f2_score(topics_test['content_ids'], topics_test['predictions'])
print(f'Our max positive score is {pos_score}')
print(f'Our f2 score is {f2_score}')
print(f'Our recall is {recall}')

# We can delete correlations
del correlations
gc.collect()

# Set id as index for content
content.set_index('id', inplace = True)

# topicsを完成させる
#full_correlations = pd.read_csv('/kaggle/input/finetuning-model-all-minilm-l6-v2/kfold_correlations.csv')
topics_full = topics.merge(full_correlations, how = 'inner', left_on = ['id'], right_on = ['topic_id'])
# 予測のところに正解データを加えるべきか、rerank時のCVスコアによって検証したい
topics_full['predictions'] = topics_full.apply(lambda x: ' '.join(list(set(x.predictions.split(' ') + x.content_ids.split(' ')))) \
                                               if x.fold != 0 else x.predictions, axis = 1)  # fold0以外にはtopicsに正解(コンテンツ)がないので入れて完成させる

# Building training set
train = build_training_set(topics_full, content, CFG)
print(f'Our training set has {len(train)} rows')

# Save train set to disk to train on another notebook
train.to_csv('top50_train_with_groundtruth_new.csv', index = False)
train.head()
