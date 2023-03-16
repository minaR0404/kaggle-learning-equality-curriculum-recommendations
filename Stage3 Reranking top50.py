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
from sklearn.model_selection import StratifiedGroupKFold
%env TOKENIZERS_PARALLELISM=true
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================================================================================
# Configurations
# =========================================================================================
class CFG:
    print_freq = 100   # 指定された時間間隔での関数呼び出しでプロファイル結果を出力
    num_workers = 4   # 複数処理 
    model = "/content/drive/MyDrive/Colab Notebooks/all-MiniLM-L6-v2-exp_batch288_fold4_stage1_epochs30"
    #sup_model_tuned = '/content/drive/MyDrive/Colab Notebooks/all-MiniLM-L6-v2-exp_batch288_fold4_bread_epochs30_fold0_42_stage2_2.pth'
    tokenizer = AutoTokenizer.from_pretrained(model)
    gradient_checkpointing = True  # VRAMの消費量を削減させて、バッチサイズを上げる
    num_cycles = 0.5   # コサインスケジュールの波の数
    warmup_ratio = 0.1   # 学習時のウォームアップ部分の割合
    epochs = 10   # エポック数
    encoder_lr = 1e-5   # エンコーダの学習率
    decoder_lr = 1e-4   # デコーダの学習率
    # AdamWのパラメータ設定
    eps = 1e-6   # 数値安定性に対するアダムのイプシロン
    betas = (0.9, 0.999)   # Adamのベータパラメータ (b1, b2)
    batch_size = 384   # バッチサイズ
    weight_decay = 0.01   # 重みの減衰(L2ペナルティ)
    max_grad_norm = 0.012   # 最大勾配クリッピング
    max_len = 256   # 文章(?)の最大長 text_len512 = topics_len256 + content_len256
    n_folds = 4   # フォールド(データの分割)数
    seed = 42   # 乱数シード値
    
# =========================================================================================
# Seed everything for deterministic results
# =========================================================================================
# シード値を固定する関数
def seed_everything(cfg):
    random.seed(cfg.seed)  # 乱数シードの設定(42)
    os.environ['PYTHONHASHSEED'] = str(cfg.seed)   # 環境変数の['PythonHashSeed']カラムを設定
    np.random.seed(cfg.seed)  # Numpyの乱数シード
    torch.manual_seed(cfg.seed)  # PyTorchの乱数シード
    torch.cuda.manual_seed(cfg.seed)  # Cudaの乱数シード
    torch.backends.cudnn.deterministic = True  # Cudaの操作内で決定的に設定する
    
# =========================================================================================
# F2 score metric
# =========================================================================================
# F2スコアを計算する関数
def f2_score(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))  # 正解データ(コンテンツID)
    y_pred = y_pred.apply(lambda x: set(x.split()))  # 予測データ
    # 考え方：予測値(正：Positive,偽：Negative)がどうだったか(合:True,違:False)
    # ちなみにx[0]は正解データ、x[1]は予測データ
    tp = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])  # True-Positive
    fp = np.array([len(x[1] - x[0]) for x in zip(y_true, y_pred)])  # False-Positive
    fn = np.array([len(x[0] - x[1]) for x in zip(y_true, y_pred)])  # False-Negative
    precision = tp / (tp + fp)  # 適合率
    recall = tp / (tp + fn)  # 再現率
    f2 = tp / (tp + 0.2 * fp + 0.8 * fn)
    return round(f2.mean(), 4)  # F2スコア

# =========================================================================================
# Data Loading
# =========================================================================================
def read_data(cfg):
    # 訓練データは別のノートブックの教師なしデータから読み込む
    #train = pd.read_csv('/kaggle/input/unsupervised-train-set/top50_train.csv') 
    #train = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/data_top50/top50_train_01.csv')
    train = pd.read_csv('/path/to/dest/top50_train_with_groundtruth_new.csv')
    #train = train[:600000]   # 実験サイクルのため一部のデータのみ使う
    # train['topics_set'] = [train['topics_set'][i][:128] for i in range(len(train))]
    # train['content_set'] = [train['content_set'][i][:128] for i in range(len(train))]
    # train['title1'].fillna("Title does not exist", inplace = True)
    # train['title2'].fillna("Title does not exist", inplace = True)
    train = train.dropna().reset_index(drop=True)
    train['fold'] = train['fold'].astype('int32')
    train['target'] = train['target'].astype('int32')
    correlations = pd.read_csv('/content/correlations.csv')
    # Create feature column
    # 特徴量カラムを作成する(=SEPトークンを加えた['text']は特徴量として使えるということ)
    #train['text'] = train['title1'] + '[SEP]' + train['title2']  # トピックとコンテンツのタイトルを[SEP]トークンで分割する
    #train['text'] = train[['topics_set', 'content_set']].values.tolist()  # リスト形式
    train['text'] = train['topics_set'] + '[SEP]' + train['content_set']
    print(' ')
    print('-' * 50)
    print(f"train.shape: {train.shape}")
    print(f"correlations.shape: {correlations.shape}")
    return train, correlations

# =========================================================================================
# Get max length
# =========================================================================================
# トークンの最大長を求める関数
def get_max_length(train, cfg):
    lengths = []
    for text in tqdm(train['text'].fillna("").values, total = len(train)):  # textがリスト形式だと、なぜかmax_len=4になって学習が止まる
        # ここでの長さとは、タイトル文章のトークナイザー(文章のトークン化 + 数値テンソル変換)のトークンIDの配列の長さのこと。
        length = len(cfg.tokenizer(text, add_special_tokens = False)['input_ids'])
        lengths.append(length)
    cfg.max_len = max(lengths) + 2 # cls & sep ← 最低限必要な特殊トークンのこと
    print(f"max_len: {cfg.max_len}")

# =========================================================================================
# Prepare input, tokenize
# =========================================================================================
# インプット(トークナイザーのテンソル型の3種の配列)を用意する関数
# inputs = [’token_id’], [’token_type_id’], [’attention_mask’]
def prepare_input(text, cfg):
    # encode_plusは、tokenizerのtoken_id, token_type_id, attention_maskを同時出力できる
    inputs = cfg.tokenizer.encode_plus(
        text,    # topics['set'], content['set'] のリスト形式で入力する(それぞれのsetは[SEP]の文章) 
        #is_split_into_words=True,  # textをリスト形式で入力した時に必要らしい
        return_tensors = None, 
        add_special_tokens = True, 
        max_length = cfg.max_len,
        pad_to_max_length = True,  # 最大長でパディング(長さを揃える)
        truncation = True  # 最大長でカットするためのオプション
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype = torch.long)  # インプットをテンソルに変換(long=int64)
    return inputs

# =========================================================================================
# Custom dataset
# =========================================================================================
# 自作データセットを作成時に参照する関数
class custom_dataset(Dataset):
    def __init__(self, df, cfg):
        self.cfg = cfg
        self.texts = df['text'].values  # ここの'text'は topics['set'], content['set'] 2つの[SEP]文章のリスト形式
        self.labels = df['target'].values
    # __len__() : クラスインスタンスにlen()を使った時に呼ばれる関数
    def __len__(self):
        return len(self.texts)
    # __getitem__() : クラスインスタンスの要素を参照するときに呼ばれる関数
    def __getitem__(self, item):
        # タイトル文章を入れてインプットを取得する
        # 要は、インプットを使ってデータセット作成している
        inputs = prepare_input(self.texts[item], self.cfg)  # texts[item] = text
        label = torch.tensor(self.labels[item], dtype = torch.float)
        return inputs, label
    
# =========================================================================================
# Collate function for training
# =========================================================================================
# inputsの中身を整える関数
def collate(inputs):
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())  # attention_maskの最大長
    for k, v in inputs.items():
        inputs[k] = inputs[k][:,:mask_len]  # inputsの中身において、最初の:は全行、最後の:はmask_lenまでの列 → 最大長以内に抑える(?)
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
# Model
# =========================================================================================
# モデルの設定
class custom_model(nn.Module):
    # 学習モデルの設定
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states = True)
        self.config.hidden_dropout = 0.0  # ドロップアウトはなし
        self.config.hidden_dropout_prob = 0.0
        self.config.attention_dropout = 0.0
        self.config.attention_probs_dropout_prob = 0.0
        self.model = AutoModel.from_pretrained(cfg.model, config = self.config)
        if self.cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.pool = MeanPooling()  # 平均プーリング
        self.fc = nn.Linear(self.config.hidden_size, 1)
        self._init_weights(self.fc)
    # 重みの初期化(あとで事前学習モデルからの重みを適用するため)
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
    # 特徴量を算出する関数
    def feature(self, inputs):
        outputs = self.model(**inputs)  # 使用モデルは xlm-roberta-base
        last_hidden_state = outputs.last_hidden_state
        feature = self.pool(last_hidden_state, inputs['attention_mask'])  # プーリングの出力結果を特徴量とする
        return feature
    # 順伝播の関数
    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(feature)  # プーリングの結果をnn.Linearモデルに投げる
        return output
    
# =========================================================================================
# Helper functions
# =========================================================================================
# ロス・正答率(?)などの損失関数を計算する関数
class AverageMeter(object):
    """Computes and stores the average and current value"""  # 平均値と最新値を計算・記録する
    def __init__(self):
        self.reset()
    # ゼロにリセットする関数
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    # 値を更新する関数
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# 時間(分・秒)を計算する関数
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# 時間の経過を計算する関数
def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))

# =========================================================================================
# Train function loop
# =========================================================================================
# モデルを学習する関数
def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device, cfg):
    model.train()  # モデルを学習
    scaler = torch.cuda.amp.GradScaler(enabled = True)  # スケーラーの定義。自動混合精度で勾配情報をスケール(乗算)するため
    losses = AverageMeter()  # 損失関数の定義
    start = end = time.time()
    global_step = 0
    # リスト形式だと、なぜかここで学習が止まっている。ので、後で検証が必要
    for step, (inputs, target) in enumerate(train_loader):  # ミニバッチをループ。訓練データの中身はinputs,target(?)
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)  # inputsの値をcpu/gpuどっちかに入れる
        target = target.to(device)
        batch_size = target.size(0)  # 0次元数(最初の次元の要素)のテンソルのサイズをバッチサイズに設定
        # 演算処理をamp(Automatic Mixed Precision)の対象にキャスト → GPUメモリの省エネ化
        with torch.cuda.amp.autocast(enabled = True):  # withは処理の一時的な有効化
            y_preds = model(inputs)  # 予測ベクトル
            loss = criterion(y_preds.view(-1), target)  # 予測ベクトルの次元、ターゲットを評価式に入れて、損失値を出力する
        losses.update(loss.item(), batch_size)  # ロスを損失関数に入れて更新する
        scaler.scale(loss).backward()  # スケールした勾配を作る(損失計算のスケール化、逆誤差伝搬)
        scaler.unscale_(optimizer)  # 勾配をアンスケール(分割)する
        # 訓練中に勾配の爆発を防ぐために、すべての勾配を一緒にスケーリングする
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)  # 勾配をクリップ(↑の意味)する
        scaler.step(optimizer)  # パラメータの更新
        scaler.update()  # スケーラーの更新
        optimizer.zero_grad()  # 最適化対象の全てのパラメータの勾配をゼロにする
        global_step += 1  # ここで1ステップが終了
        scheduler.step()  # スケジューラーを使って、エポックごとに学習率を変化させる
        end = time.time()
        # 以下、学習状況を表示する
        if step % cfg.print_freq == 0 or step == (len(train_loader) - 1):  # ステップが区切り(500)で表示(% は余り)
            print('Epoch: [{0}][{1}/{2}] '  # エポック数、1エポックのステップ状況
                  'Elapsed {remain:s} '  # 経過時間
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '  # ロス(損失値、損失平均)
                  'Grad: {grad_norm:.4f}  '  # 勾配情報
                  'LR: {lr:.8f}  '  # 学習率
                  .format(epoch + 1, 
                          step,  # 現在のステップ(バッチ数?)
                          len(train_loader),  # データ総数
                          remain = timeSince(start, float(step + 1) / len(train_loader)),
                          loss = losses,
                          grad_norm = grad_norm,
                          lr = scheduler.get_lr()[0]))
    # 最終的に出力されるのは、損失関数の平均値である
    return losses.avg

# =========================================================================================
# Valid function loop
# =========================================================================================
# 評価データから損失関数とシグモイド関数を算出する関数
def valid_fn(valid_loader, model, criterion, device, cfg):
    losses = AverageMeter()  # 損失関数
    model.eval()  # モデルを式として評価
    preds = []
    start = end = time.time()
    for step, (inputs, target) in enumerate(valid_loader):  # 評価データのミニバッチループ
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        target = target.to(device)
        batch_size = target.size(0)
        with torch.no_grad():   # テンソルの勾配計算を無効化 
            y_preds = model(inputs)  # モデルからベクトル出力
        loss = criterion(y_preds.view(-1), target)  # ロス算出
        losses.update(loss.item(), batch_size)  # 損失関数の更新
        preds.append(y_preds.sigmoid().squeeze().to('cpu').numpy().reshape(-1))  # ベクトルのシグモイド関数を予測結果とする
        end = time.time()
        if step % cfg.print_freq == 0 or step == (len(valid_loader) - 1):
            print('EVAL: [{0}/{1}] '  # 評価データ中のステップ数
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(step, 
                          len(valid_loader),
                          loss = losses,
                          remain = timeSince(start, float(step + 1) / len(valid_loader))))
    predictions = np.concatenate(preds, axis = 0)  # 予測結果を1つにまとめる
    # 損失関数の平均値、予測結果を返す
    return losses.avg, predictions

# =========================================================================================
# Get best threshold
# =========================================================================================
# F2スコアを最大化する最適な閾値を求める関数
def get_best_threshold(x_val, val_predictions, correlations):
    best_score = 0   # ベストスコア
    best_threshold = None  # 閾値(シグモイド関数)
    for thres in np.arange(0.001, 0.1, 0.001):   # 0.001~0.1の範囲で閾値を動かす
        # 評価データの予測シグモイドが閾値より高いトピック・コンテンツのペアのみ抽出して'1'を与え、それ以外は'0'を与える
        x_val['predictions'] = np.where(val_predictions > thres, 1, 0)
        x_val1 = x_val[x_val['predictions'] == 1]  # シグモイド関数が1
        x_val1 = x_val1.groupby(['topics_ids'])['content_ids'].unique().reset_index()  # トピックごとにコンテンツを整理する
        x_val1['content_ids'] = x_val1['content_ids'].apply(lambda x: ' '.join(x))  # コンテンツをまとめる
        x_val1.columns = ['topic_id', 'predictions']  # トピックIDと予測(コンテンツの集合)にまとめる
        x_val0 = pd.Series(x_val['topics_ids'].unique())
        x_val0 = x_val0[~x_val0.isin(x_val1['topic_id'])]  # 閾値を超えたコンテンツが皆無だったトピックIDを抽出する
        x_val0 = pd.DataFrame({'topic_id': x_val0.values, 'predictions': ""})  # そのトピックには予測(コンテンツ)を入れない
        x_val_r = pd.concat([x_val1, x_val0], axis = 0, ignore_index = True)  # 全てのトピックをまとめる
        x_val_r = x_val_r.merge(correlations, how = 'left', on = 'topic_id')  # corr.dfともまとめる(正解データ:正解のペアのdf)
        score = f2_score(x_val_r['content_ids'], x_val_r['predictions'])  # コンテンツについて、予測と正解データをF2スコアの計算にかける
        if score > best_score:
            best_score = score  # F2スコアの更新
            best_threshold = thres  # 最適な閾値の更新
    # F2スコアが最も高くなった時の、閾値とF2スコアを返す
    return best_score, best_threshold
    
# =========================================================================================
# Train & Evaluate
# =========================================================================================
# 1フォールドの学習・評価ループを行うメイン関数
def train_and_evaluate_one_fold(train, correlations, fold, cfg):
    print(' ')
    print(f"========== fold: {fold} training ==========")
    # Split train & validation
    x_train = train[train['fold'] != fold]  # フォールド以外を訓練データ
    x_val = train[train['fold'] == fold]  # フォールドを評価データ
    valid_labels = x_val['target'].values
    train_dataset = custom_dataset(x_train, cfg)  # データセットの中身はinputs,label
    valid_dataset = custom_dataset(x_val, cfg)
    train_loader = DataLoader(   # ミニバッチ作成
        train_dataset, 
        batch_size = cfg.batch_size, 
        shuffle = True, 
        num_workers = cfg.num_workers,  # 複数処理
        pin_memory = True,  # 高速化
        drop_last = True  # 最後にドロップ
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size = cfg.batch_size, 
        shuffle = False, 
        num_workers = cfg.num_workers, 
        pin_memory = True, 
        drop_last = False
    )
    # Get model
    model = custom_model(cfg)
    model.to(device)
    # Load weights
    state = torch.load(cfg.sup_model_tuned)
    model.load_state_dict(state['model'])
    # Optimizer
    # エンコーダー・デコーダーの学習率・重み減衰のパラメータを最適化する関数
    def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay = 0.0):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']  # 減衰を無効化するオプション
        optimizer_parameters = [
            # モデルのパラメータ内に減衰オプションがない場合
            {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
            'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],  # 減衰オプションありの場合
            'lr': encoder_lr, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if "model" not in n],  # モデルがない場合
            'lr': decoder_lr, 'weight_decay': 0.0}
        ]
        return optimizer_parameters
    optimizer_parameters = get_optimizer_params(
        model, 
        encoder_lr = cfg.encoder_lr, 
        decoder_lr = cfg.decoder_lr,
        weight_decay = cfg.weight_decay
    )
    optimizer = AdamW(   # AdamWの最適化
        optimizer_parameters, 
        lr = cfg.encoder_lr, 
        eps = cfg.eps,  # イプシロンパラメータ
        betas = cfg.betas  # ベータパラメータ
    )
    num_train_steps = int(len(x_train) / cfg.batch_size * cfg.epochs)  # エポック分、ステップ数も多くなる
    num_warmup_steps = num_train_steps * cfg.warmup_ratio  # ウォームアップ(捨てる期間)
    # Scheduler
    scheduler = get_cosine_schedule_with_warmup(  # cosine波形で減衰するScheduler(学習率変化器)を作成
        optimizer, 
        num_warmup_steps = num_warmup_steps, 
        num_training_steps = num_train_steps, 
        num_cycles = cfg.num_cycles
        )
    # Training & Validation loop
    # 学習・評価ループ
    # 訓練・評価データから損失関数・シグモイド関数を得てF2スコアを最大化し、最適な重み情報をモデルとして保存する
    criterion = nn.BCEWithLogitsLoss(reduction = "mean")  # バイナリクロスエントロピー損失関数(reduction: 各ミニバッチの観測値でロスを平均化)
    best_score = 0
    for epoch in range(cfg.epochs):  # エポックごとにループ
        start_time = time.time()
        # Train
        avg_loss = train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device, cfg)
        # Validation
        avg_val_loss, predictions = valid_fn(valid_loader, model, criterion, device, cfg)
        # Compute f2_score
        score, threshold = get_best_threshold(x_val, predictions, correlations)
        elapsed = time.time() - start_time
        print(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        print(f'Epoch {epoch+1} - Score: {score:.4f} - Threshold: {threshold:.5f}')
        if score > best_score:
            best_score = score
            print(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save(
                {'model': model.state_dict(), 'predictions': predictions}, 
                f"/content/drive/My Drive/Colab Notebooks/{cfg.model.replace('/', '-')}_fold{fold}_{cfg.seed}_stage2_3.pth"
                )
            val_predictions = predictions
    torch.cuda.empty_cache()
    gc.collect()
    # Get best threshold
    best_score, best_threshold = get_best_threshold(x_val, val_predictions, correlations)
    print(f'Our CV score is {best_score} using a threshold of {best_threshold}')
    
# 以下、メイン処理
# Seed everything
seed_everything(CFG)
# Read data
train, correlations = read_data(CFG)
# CV split
#cv_split(train, CFG)
# Get max length
get_max_length(train, CFG)
# Train and evaluate one fold
train_and_evaluate_one_fold(train, correlations, 0, CFG)
