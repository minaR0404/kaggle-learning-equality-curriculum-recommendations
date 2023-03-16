# =================================================================================
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
from pathlib import Path
%env TOKENIZERS_PARALLELISM=false
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pd.set_option('display.max_rows', None)
threshold = 0.06
top_n = 50

class CFG1:
    uns_model = "/kaggle/input/finetuning-model-11/all-MiniLM-L6-v2-exp_batch288_fold4_bread_epochs30"
    sup_model = "/kaggle/input/finetuning-model-11/all-MiniLM-L6-v2-exp_batch288_fold4_bread_epochs30"
    sup_model_tuned = "/kaggle/input/stage2-pth-1/all-MiniLM-L6-v2-exp_batch288_fold4_bread_epochs30_fold0_42_stage2_2.pth"
    uns_tokenizer = AutoTokenizer.from_pretrained(uns_model)  # + '/tokenizer'
    sup_tokenizer = AutoTokenizer.from_pretrained(sup_model)
    pooling = "mean"
    batch_size = 16
    gradient_checkpointing = False   # False(default)
    add_with_best_prob = False   # False
    
class CFG2:
    uns_model = "/kaggle/input/stage-1-paraphrase-multilingual-mpnet-base-v2-4747/stage-1-paraphrase-multilingual-mpnet-base-v2-tuned-4747"
    sup_model = "/kaggle/input/paraphrasemultilingualmpnetbasev2-origin2/paraphrasemultilingualmpnetbasev2-origin"
    sup_model_tuned = "/kaggle/input/paraphrase-multilingual-mpnet-base-v2-reranker/model-paraphrase-multilingual-mpnet-base-v2-tuned_0.4747.pth"
    uns_tokenizer = AutoTokenizer.from_pretrained(uns_model + '/tokenizer')
    sup_tokenizer = AutoTokenizer.from_pretrained(sup_model + '/tokenizer')
    pooling = "mean"
    batch_size = 120
    gradient_checkpointing = False
    add_with_best_prob = True  

CFG_list = [CFG1]  # [CFG1, CFG2]


# =================================================================================
# Read data
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
    # Merge topics with sample submission to only infer test topics
    topics = topics_df.merge(sample_submission, how = 'inner', left_on = 'id', right_on = 'topic_id').set_index('id')
    topics_df = topics_df.set_index('id')

    topic_id_texts = []
    content_id_texts = []
    for topic_idx in tqdm(topics.index):   # topics.idx
        tmp_topic = Topic(topic_idx)
        children = tmp_topic.children
        child = "" if len(children)==0 else children[0].description

        parent = tmp_topic.parent
        par = "" if parent is None else parent.description
        # [SEP]{tmp_topic.description}[SEP]{tmp_topic.get_breadcrumbs()} を抜いてる
        topic_repre = f"{tmp_topic.title}[SEP]{tmp_topic.description}[SEP]{tmp_topic.get_breadcrumbs()}"[:128]  # CLS,SEP 入れる必要ある(?)

        topic_id_texts.append((topic_idx, topic_repre))

    for content_idx in tqdm(content_df.index):
        ct = ContentItem(content_idx)
        # [SEP]{ct.description}[SEP]{ct.text}　を抜いてる
        content_repre = f"{ct.title}[SEP]{ct.description}[SEP]{ct.text}"[:128]
        content_id_texts.append((content_idx, content_repre))

    topics = pd.DataFrame(data={'id':[item[0] for item in topic_id_texts], 
                             'title':[item[1] for item in topic_id_texts]})
    content = pd.DataFrame(data={'id':[item[0] for item in content_id_texts], 
                                 'title':[item[1] for item in content_id_texts]})

    del topics_df, content_df, sample_submission, topic_id_texts, content_id_texts
    gc.collect()

    print(' ')
    print('-' * 50)
    print(f"topics.shape: {topics.shape}")
    print(f"content.shape: {content.shape}")

    
    return topics, content
  

def prepare_uns_input(text, cfg):
    inputs = cfg.uns_tokenizer.encode_plus(
        text, 
        max_length = 128,
        truncation=True,
        return_tensors = None, 
        add_special_tokens = True, 
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype = torch.long)
    return inputs


# =========================================================================================
# pooling class
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

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, last_hidden_state, attention_mask):
        w = self.attention(last_hidden_state).float()
        w[attention_mask==0]=float('-inf')
        w = torch.softmax(w,1)
        attention_embeddings = torch.sum(w * last_hidden_state, dim=1)
        return attention_embeddings

class MaxPooling(nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()       
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = -1e4
        max_embeddings, _ = torch.max(embeddings, dim = 1)
        return max_embeddings
    
    
class MinPooling(nn.Module):
    def __init__(self):
        super(MinPooling, self).__init__()     
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = 1e-4
        min_embeddings, _ = torch.min(embeddings, dim = 1)
        return min_embeddings
    

class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights = None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
                torch.tensor([1] * (num_hidden_layers+1 - layer_start), dtype=torch.float)
            )

    def forward(self, features):
        ft_all_layers = features['all_layer_embeddings']

        all_layer_embedding = torch.stack(ft_all_layers)
        all_layer_embedding = all_layer_embedding[self.layer_start:, :, :, :]

        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor*all_layer_embedding).sum(dim=0) / self.layer_weights.sum()

        features.update({'token_embeddings': weighted_average})
        return features
    
class ConcatPooling(nn.Module):
    def __init__(self, backbone_config, pooling_config):
        super(ConcatPooling, self, ).__init__()

        self.n_layers = pooling_config.n_layers
        self.output_dim = backbone_config.hidden_size*pooling_config.n_layers

    def forward(self, inputs, backbone_outputs):
        all_hidden_states = get_all_hidden_states(backbone_outputs)

        concatenate_pooling = torch.cat([all_hidden_states[-(i + 1)] for i in range(self.n_layers)], -1)
        concatenate_pooling = concatenate_pooling[:, 0]
        return concatenate_pooling

# =========================================================================================
# Unsupervised dataset
# =========================================================================================
class uns_dataset(Dataset):
    def __init__(self, df, cfg):
        self.cfg = cfg
        self.texts = df['title'].values
        #self.texts = df['set'].values
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, item):
        inputs = prepare_uns_input(self.texts[item], self.cfg)
        return inputs

# =========================================================================================
# Prepare input, tokenize
# =========================================================================================
def prepare_sup_input(text, cfg):
    inputs = cfg.sup_tokenizer.encode_plus(
        text, 
        max_length = 256,
        truncation=True,
        return_tensors = None, 
        add_special_tokens = True, 
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype = torch.long)
    return inputs

# =========================================================================================
# Supervised dataset
# =========================================================================================
class sup_dataset(Dataset):
    def __init__(self, df, cfg):
        self.cfg = cfg
        self.texts = df['text'].values
        #self.texts = df['test_set'].values
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, item):
        inputs = prepare_sup_input(self.texts[item], self.cfg)
        return inputs

# =========================================================================================
# Unsupervised model
# =========================================================================================
class uns_model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg == CFG1:
            self.config = AutoConfig.from_pretrained(cfg.uns_model)
            self.model = AutoModel.from_pretrained(cfg.uns_model, config = self.config)
        else:
            self.config = AutoConfig.from_pretrained(cfg.uns_model + '/config')
            self.model = AutoModel.from_pretrained(cfg.uns_model + '/model', config = self.config)
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
def get_pos_socre(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
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
# Build our inference set
# =========================================================================================
def build_inference_set(topics, content, cfg):
    # Create lists for training
    topics_ids = []
    content_ids = []
    title1 = []
    title2 = []
    # Iterate over each topic
    for k in tqdm(range(len(topics))):
        row = topics.iloc[k]
        topics_id = row['id']
        topics_title = row['title']
        #topics_feature = row['set']  # set = title + description + breadcrumbs
        predictions = row['predictions'].split(' ')
        for pred in predictions:
            content_title = content.loc[pred, 'title']
            topics_ids.append(topics_id)
            content_ids.append(pred)
            title1.append(topics_title)
            title2.append(content_title)
            
    # Build training dataset
    test = pd.DataFrame(
        {'topics_ids': topics_ids, 
         'content_ids': content_ids, 
         'title1': title1, 
         'title2': title2,
        }
    )
    # Release memory
    del topics_ids, content_ids, title1, title2
    gc.collect()
    
    return test
    
# =========================================================================================
# Get neighbors
# =========================================================================================
def get_neighbors(tmp_topics, tmp_content, cfg):
    # Create topics dataset
    topics_dataset = uns_dataset(tmp_topics, cfg)
    # Create content dataset
    content_dataset = uns_dataset(tmp_content, cfg)
    # Create topics and content dataloaders
    topics_loader = DataLoader(
        topics_dataset, 
        batch_size = cfg.batch_size, 
        shuffle = False, 
        collate_fn = DataCollatorWithPadding(tokenizer = cfg.uns_tokenizer, padding = 'longest'),
        num_workers = 4, 
        pin_memory = True, 
        drop_last = False
    )
    content_loader = DataLoader(
        content_dataset, 
        batch_size = cfg.batch_size, 
        shuffle = False, 
        collate_fn = DataCollatorWithPadding(tokenizer = cfg.uns_tokenizer, padding = 'longest'),
        num_workers = 4, 
        pin_memory = True, 
        drop_last = False
        )
    # Create unsupervised model to extract embeddings
    model = uns_model(cfg)
    model.to(device)
    # Predict topics
    topics_preds = get_embeddings(topics_loader, model, device)
    content_preds = get_embeddings(content_loader, model, device)
    # Transfer predictions to gpu
    topics_preds_gpu = cp.array(topics_preds)
    content_preds_gpu = cp.array(content_preds)
    # Release memory
    del topics_dataset, content_dataset, topics_loader, content_loader, topics_preds, content_preds
    gc.collect()
    torch.cuda.empty_cache()
    # KNN model
    print(' ')
    print('Training KNN model...')
    neighbors_model = NearestNeighbors(n_neighbors = top_n, metric = 'cosine')
    neighbors_model.fit(content_preds_gpu)
    indices = neighbors_model.kneighbors(topics_preds_gpu, return_distance = False)
    predictions = []
    for k in range(len(indices)):
        pred = indices[k]
        p = ' '.join([tmp_content.loc[ind, 'id'] for ind in pred.get()])
        predictions.append(p)
    tmp_topics['predictions'] = predictions
    # Release memory
    del topics_preds_gpu, content_preds_gpu, neighbors_model, predictions, indices, model
    gc.collect()
    torch.cuda.empty_cache()
    return tmp_topics, tmp_content

# =========================================================================================
# Process test
# =========================================================================================
def preprocess_test(tmp_test):
    tmp_test['title1'].fillna("Title does not exist", inplace = True)
    tmp_test['title2'].fillna("Title does not exist", inplace = True)
    # Create feature column
    tmp_test['text'] = tmp_test['title1'] + '[SEP]' + tmp_test['title2']
    #tmp_test['test_set'] = tmp_test['topics_set'] + '[SEP]' + tmp_test['content_set']
    # Drop titles
    tmp_test.drop(['title1', 'title2'], axis = 1, inplace = True)  #, 'topics_set', 'content_set'
    # Sort so inference is faster
    tmp_test['length'] = tmp_test['text'].apply(lambda x: len(x))
    tmp_test.sort_values('length', inplace = True)
    tmp_test.drop(['length'], axis = 1, inplace = True)
    tmp_test.reset_index(drop = True, inplace = True)
    gc.collect()
    torch.cuda.empty_cache()
    return tmp_test

# =========================================================================================
# Model
# =========================================================================================
class custom_model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg == CFG1:
            self.config = AutoConfig.from_pretrained(cfg.sup_model, output_hidden_states = True)
        else:
            self.config = AutoConfig.from_pretrained(cfg.sup_model + '/config', output_hidden_states = True)
        self.config.hidden_dropout = 0.0
        self.config.hidden_dropout_prob = 0.0
        self.config.attention_dropout = 0.0
        self.config.attention_probs_dropout_prob = 0.0
        if cfg == CFG1:
            self.model = AutoModel.from_pretrained(cfg.sup_model, config = self.config)
        else:
            self.model = AutoModel.from_pretrained(cfg.sup_model + '/model', config = self.config)
        #self.pool = MeanPooling()
        if self.cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        if CFG.pooling == 'mean' or CFG.pooling == "ConcatPool":
            self.pool = MeanPooling()
        elif CFG.pooling == 'max':
            self.pool = MaxPooling()
        elif CFG.pooling == 'min':
            self.pool = MinPooling()
        elif CFG.pooling == 'attention':
            self.pool = AttentionPooling(self.config.hidden_size)
        elif CFG.pooling == "WLP":
            self.pool = WeightedLayerPooling(self.config.num_hidden_layers, layer_start=6)
        
        if CFG.pooling == "ConcatPool":
            self.fc = nn.Linear(self.config.hidden_size*4, 1)  
        else:
            self.fc = nn.Linear(self.config.hidden_size, 1)
        #self.fc = nn.Linear(self.config.hidden_size, 1)
        self._init_weights(self.fc)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    def feature(self, inputs):
        outputs = self.model(**inputs)
        
        if CFG.pooling == "WLP":
            last_hidden_state = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            tmp = {
                'all_layer_embeddings': last_hidden_state.hidden_states
            }
            feature = self.pool(tmp)['token_embeddings'][:, 0]
            
        elif CFG.pooling == "ConcatPool":
            last_hidden_state = torch.stack(self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask']).hidden_states)

            p1 = self.pool(last_hidden_state[-1], inputs['attention_mask'])
            p2 = self.pool(last_hidden_state[-2], inputs['attention_mask'])
            p3 = self.pool(last_hidden_state[-3], inputs['attention_mask'])
            p4 = self.pool(last_hidden_state[-4], inputs['attention_mask'])

            feature = torch.cat(
                (p1, p2, p3, p4),-1
            )
               
        else:
            last_hidden_state = outputs.last_hidden_state
            feature = self.pool(last_hidden_state, inputs['attention_mask'])
        
        #last_hidden_state = outputs.last_hidden_state
        #feature = self.pool(last_hidden_state, inputs['attention_mask'])
        return feature
    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(feature)
        return output
# =========================================================================================
# Inference function loop
# =========================================================================================
def inference_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total = len(test_loader))
    for inputs in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds.sigmoid().squeeze().to('cpu').numpy().reshape(-1))
    predictions = np.concatenate(preds)
    return predictions

# =========================================================================================
# Inference
# =========================================================================================
def inference(test, cfg, _idx):
    # Create dataset and loader
    test_dataset = sup_dataset(test, cfg)
    test_loader = DataLoader(
        test_dataset, 
        batch_size = cfg.batch_size, 
        shuffle = False, 
        collate_fn = DataCollatorWithPadding(tokenizer = cfg.sup_tokenizer, padding = 'longest'),
        num_workers = 8,
        pin_memory = True,
        drop_last = False
    )
    # Get model
    model = custom_model(cfg)
    
    # Load weights
    state = torch.load(cfg.sup_model_tuned, map_location = torch.device('cpu'))
    model.load_state_dict(state['model'])
    prediction = inference_fn(test_loader, model, device)
    
    # Release memory
    torch.cuda.empty_cache()
    del test_dataset, test_loader, model, state
    gc.collect()
    
    # Use threshold
    test['probs'] = prediction
    test['predictions'] = test['probs'].apply(lambda x: int(x > threshold))
    test = test.merge(test.groupby("topics_ids", as_index=False)["probs"].max(), on="topics_ids", suffixes=["", "_max"])
    #test = test[test['has_contents'] == True]  # has_content = True のtopicsは全体の8割
    #display(test)
    
    test1 = test[test['predictions'] == 1]  # & (test['topic_language'] == test['content_language'])]
    test1 = test1.groupby(['topics_ids'])['content_ids'].unique().reset_index()
    test1['content_ids'] = test1['content_ids'].apply(lambda x: ' '.join(x))
    test1.columns = ['topic_id', 'content_ids']
    #display(test1.head())
    
    test0 = pd.Series(test['topics_ids'].unique())
    test0 = test0[~test0.isin(test1['topic_id'])]
    test0 = pd.DataFrame({'topic_id': test0.values, 'content_ids': ""})
    if cfg.add_with_best_prob:
        test0 = test0[["topic_id"]].merge(test[test['probs'] == test['probs_max']][["topics_ids", "content_ids"]],
                                          left_on="topic_id", right_on="topics_ids")[['topic_id', "content_ids"]]
    #display(test0.head())
    test_r = pd.concat([test1, test0], axis = 0, ignore_index = True)
    test_r.to_csv(f'submission_{_idx+1}.csv', index = False)
    
    return test_r
  
  
  for _idx, CFG in enumerate(CFG_list):
    # Read data
    tmp_topics, tmp_content = read_data(CFG)
    # Run nearest neighbors
    tmp_topics, tmp_content = get_neighbors(tmp_topics, tmp_content, CFG)
    gc.collect()
    torch.cuda.empty_cache()
    # Set id as index for content
    tmp_content.set_index('id', inplace = True)
    # Build training set
    tmp_test = build_inference_set(tmp_topics, tmp_content, CFG)
    # Process test set
    tmp_test = preprocess_test(tmp_test)
    # Inference
    inference(tmp_test, CFG, _idx)
    del tmp_topics, tmp_content, tmp_test
    gc.collect()
    torch.cuda.empty_cache()
    
df_test = pd.concat([pd.read_csv(f'submission_{_idx + 1}.csv') for _idx in range(len(CFG_list))])
df_test.fillna("", inplace = True)
df_test['content_ids'] = df_test['content_ids'].apply(lambda c: c.split(' '))
df_test = df_test.explode('content_ids').groupby(['topic_id'])['content_ids'].unique().reset_index()
df_test['content_ids'] = df_test['content_ids'].apply(lambda c: ' '.join(c))

#f2_test = df_test.merge(correlations, how = 'inner', left_on = ['id'], right_on = ['topic_id'])
#f2_score = f2_score(f2_test['content_ids'], f2_test['predictions'])

df_test.to_csv('submission.csv', index = False)
df_test.head()
