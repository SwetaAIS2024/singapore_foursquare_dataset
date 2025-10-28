import pandas as pd
import numpy as np
import random
import math
import json

import torch
import torch.nn.functional as F 
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Tuple


class FocalLoss(nn.Module):
    """
    多类单标签 focal loss.
    logits: [B, C]
    target: [B]  (每个样本是 0..C-1 的类别id)

    alpha:
        - None: 不做类别权重
        - Tensor[C]: class-wise 权重 (类似 class_weights)
    gamma:
        - 聚焦系数，常用 2.0
    reduction:
        - 'mean' / 'sum' / 'none'
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            # alpha 期望是 1D tensor, shape [C]
            self.register_buffer('alpha', alpha.float())
        else:
            self.alpha = None

    def forward(self, logits, target):
        # logits: [B, C]
        # target: [B]
        log_probs = F.log_softmax(logits, dim=1)        # [B, C]
        probs = log_probs.exp()                         # [B, C]

        # 取出每个样本对应真实类的 log_pt 和 pt
        # gather会根据 target 选出正确类别那一列
        target = target.long()
        log_pt = log_probs.gather(dim=1, index=target.unsqueeze(1)).squeeze(1)  # [B]
        pt = probs.gather(dim=1, index=target.unsqueeze(1)).squeeze(1)          # [B]

        # focal term
        focal_term = (1 - pt) ** self.gamma           # [B]

        # alpha term (class weight if provided)
        if self.alpha is not None:
            at = self.alpha.gather(dim=0, index=target)  # [B]
            loss = - at * focal_term * log_pt
        else:
            loss = - focal_term * log_pt                # [B]

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # 'none'



def read_dataset(data_path, poi_path, min_txn_length=4):
    poi_dict, user_dict = {}, {}
    cat2idx = {}
    user2idx = {}
    user_trajs = defaultdict(dict)
    user_view_trajs, user_txn_trajs, user_review_trajs = [], [], []
    poi_rating = defaultdict(list)

    ### processing POI data
    with open(poi_path, 'r', encoding='utf-8') as f:
        poi_data = json.load(f)   
        
    for item in poi_data:
        poi_id = item["poiId"]
        cat = item["poiCategories"][0] if item["poiCategories"] else "Unknown"
        lat = item["userLocation"]["latitude"]
        lon = item["userLocation"]["longitude"]

        if cat not in cat2idx:
            cat2idx[cat] = len(cat2idx)

        if poi_id not in poi_dict:
            poi_dict[poi_id] = {
                "idx": len(poi_dict) + 1,  # 0 reserved for padding
                "lat": float(lat),
                "lon": float(lon),
                "cat": cat2idx[cat],
                "rating": 3
            }

    
    ### processing user data
    with open(data_path, "r", encoding="utf-8") as f:
        users_record = json.load(f)
    
    gender_map = {"male": 0, "female": 1, "other": 2, None: 2}
    device_map = {"iOS": 0, "Android": 1, "other": 2, None: 2}
    referrer_map = {'friend': 0, 'ad': 1, 'map': 2, 'search': 3}
    
    for user_record in users_record:

        inter = user_record.get("interaction", {}) or {}
        
        txn = inter.get("transactions", [])
        if len(txn) < min_txn_length:
            continue
        
         # ---- 用户索引 & 画像 ----
        u = user_record['user']
        user_id_raw = str(u.get("userId", "unknown"))
        if user_id_raw not in user2idx:
            user2idx[user_id_raw] = len(user2idx)
        uidx = user2idx[user_id_raw]

        device_platform = (u.get("device") or {}).get("platform")
        device_idx = device_map.get(device_platform, 2)
        
        
        if uidx not in user_dict:
            user_dict[uidx] = {
                "user_id": uidx,
                "age": int(u.get("age") or random.randint(18, 70)),
                "gender": gender_map.get(u.get("gender"), 2),
                "device": device_idx,
                "city": ((u.get("location") or {}).get("city")) or "",
                "country": ((u.get("location") or {}).get("country")) or "",
                "app_version": ((u.get("device") or {}).get("appVersion")) or "",
            }

        # view trajectories
        view_events = []
        for v in inter.get("views", []) or []:
            poi_id = v.get("poiId")
            cats = v.get("poiCategories") or []
            cat_name = cats[0] if cats else None

            cat_idx = cat2idx[cat_name]
            lat = float(v['userLocation']["latitude"])  ## same as POI lat lon
            lon = float(v['userLocation']["longitude"])
            
            view_events.append({
                "poi": poi_id,
                "cat": cat_idx,
                "lat": lat,
                "lon": lon,
                "time": v.get("timestamp"),
                "duration": v.get("duration"),
                "referrer": referrer_map[v.get('referrer')],
                "type": "view",
            })
            
        view_events = [e for e in view_events if e.get("time")]
        view_events.sort(key=lambda e: parse_ts(e["time"]))
        if view_events:
            user_view_trajs.append([uidx, view_events])
            user_trajs[uidx]['views'] = view_events

        # Transactions
        txn_events = []
        for t in inter.get("transactions", []) or []:
            poi_id = t.get("poiId")
            cats = t.get("poiCategories") or []
            cat_name = cats[0] if cats else None

            cat_idx = cat2idx[cat_name]
            lat = float(v['userLocation']["latitude"])
            lon = float(v['userLocation']["longitude"])

            txn_events.append({
                "poi": poi_id,
                "cat": cat_idx,
                "lat": lat,
                "lon": lon,
                "time": t.get("timestamp"),
                "amount": float(t.get("amount")),
                "currency": t.get("currency"),
                "payment": t.get("paymentMethod"),
                "planning_area": t.get("planning_area"),
                "transaction_id": t.get("transactionId"),
                "type": "transaction",
            })
            
        txn_events = [e for e in txn_events if e.get("time")]
        txn_events.sort(key=lambda e: parse_ts(e["time"]))
        if txn_events:
            user_txn_trajs.append([uidx, txn_events])
            user_trajs[uidx]['transactions'] = txn_events
            

        # Reviews
        review_events = []
        for r in inter.get("reviews", []) or []:
            poi_id = r.get("poiId")
            cats = r.get("poiCategories") or []
            cat_name = cats[0] if cats else None

            cat_idx = cat2idx[cat_name]
            lat = float(v['userLocation']["latitude"])  
            lon = float(v['userLocation']["longitude"])

            review_events.append({
                "poi": poi_id,
                "cat": cat_idx,
                "lat": lat,
                "lon": lon,
                "time": r.get("timestamp"),
                "user_review_rating": float(r.get("rating")),
                "reviewText": r.get("reviewText"),
                "type": "review",
            })
            
            poi_rating[poi_id].append(float(r.get("rating")))
            
        review_events = [e for e in review_events if e.get("time")]
        review_events.sort(key=lambda e: parse_ts(e["time"]))
        if review_events:
            user_review_trajs.append([uidx, review_events])
            user_trajs[uidx]['reviews'] = review_events
        
        for poi_idx, rate_li in poi_rating.items():
            poi_dict[poi_idx]['rating'] = np.mean(rate_li)

    # print(user_trajs)
        
    print(f"user view trajs num {len(user_view_trajs)}, user transactions trajs num {len(user_txn_trajs)},\
            user review trajs num {len(user_review_trajs)}, poi num {len(poi_dict)}, user num {len(user_dict)}")
    
    
    return user_trajs, poi_dict, user_dict, cat2idx



def parse_ts(ts: str) -> datetime:
    # 兼容 ISO8601 带Z
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def process_user_info(user_dict):
    user_cat_dims = {'user_id': len(user_dict),
                     'gender':3, 
                     'device':3}
    
    all_ages = [user['age'] for user in user_dict.values()]
    age_scaler = MinMaxScaler()
    age_scaler.fit(np.array(all_ages).reshape(-1, 1))
    
    user_cont_normalizers = {'age': age_scaler}
    
    return user_cat_dims, user_cont_normalizers


def process_poi_info(poi_dict):
    
    all_cats = [poi['cat'] for poi in poi_dict.values()]
    poi_cat_dims = {'cat': max(all_cats)+1,  # +1 for padding
                    'idx': len(poi_dict)}
    
    all_lat = [poi['lat'] for poi in poi_dict.values()]
    all_lon = [poi['lon'] for poi in poi_dict.values()]
    all_rating = [poi['rating'] for poi in poi_dict.values()]
    
    poi_continuous_normalizers = {
    'lat': MinMaxScaler().fit(np.array(all_lat).reshape(-1, 1)),
    'lon': MinMaxScaler().fit(np.array(all_lon).reshape(-1, 1)),
    'rating': MinMaxScaler().fit(np.array(all_rating).reshape(-1, 1))
    }
    
    return poi_cat_dims, poi_continuous_normalizers



def process_user_histories(
    user_histories: Dict[str, Dict[str, List[dict]]],
    train_ratio: float,
    val_ratio: float,
    min_txn_len: int = 4,
) -> Tuple[List[Tuple[str, dict]], List[Tuple[str, dict]], List[Tuple[str, dict]]]:
    """
    基于每个用户的 transactions 长度做切分；对每个输出样本，都附带
    截断到 cutoff 的 views/reviews 子序列（cutoff=该样本最后一笔 transaction 的时间）。
    
    Args:
        user_histories: {user_id: {"views":[...], "transactions":[...], "reviews":[...]}}
                        三类序列已按时间排序（若未排序会在内部按时间升序处理）
        train_ratio:    训练集比例（基于 transactions 数量）
        val_ratio:      验证集比例（基于 transactions 数量）
        min_txn_len:    参与划分的最小 transactions 数（小于该值则跳过该用户）

    Returns:
        train_set, val_set, test_set
        - train_set: [(user_id, {"transactions": [...], "views": [...], "reviews": [...]})]
                     —— 对应整个训练 transactions 序列与其 cutoff 截断后的 views/reviews
        - val_set:   [(user_id, {"transactions": train+val, "views": [...], "reviews": [...]})]
                     —— 单样本（与原代码一致），history=训练+验证全部
        - test_set:  [(user_id, {"transactions": train+val+test[:i+1], "views": [...], "reviews": [...]})]
                     —— 逐步增长（每一条 test 目标都给一个样本）
    """
    train_set, val_set, test_set = [], [], []
    cat_counter = Counter()
    
    def _ensure_sorted(seq: List[dict]) -> List[dict]:
        return sorted([e for e in seq if e.get("time")], key=lambda x: parse_ts(x["time"]))
    
    def _filter_by_cutoff(seq: List[dict], cutoff_dt: datetime) -> List[dict]:
        return [e for e in seq if parse_ts(e["time"]) <= cutoff_dt]

    for user_id, seqs in user_histories.items():
        views = _ensure_sorted(seqs.get("views", []) or [])
        txns  = _ensure_sorted(seqs.get("transactions", []) or [])
        revs  = _ensure_sorted(seqs.get("reviews", []) or [])

        n = len(txns)
        if n < min_txn_len:
            continue

        # ----- 基于 transactions 计算切分位置 -----
        train_end = int(n * train_ratio)
        val_end   = int(n * (train_ratio + val_ratio))

        # 防御性调整：至少留 1 val 和 1 test
        if train_end >= n - 2:
            train_end = n - 2
        if val_end <= train_end:
            val_end = train_end + 1
        if val_end >= n:
            val_end = n - 1

        train_seq_txn = txns[:train_end]
        val_seq_txn   = txns[train_end:val_end]
        test_seq_txn  = txns[val_end:]

        # ===== 1) 训练集：一个样本，transactions=train_seq_txn =====
        if len(train_seq_txn) > 1:
            cutoff_train = parse_ts(train_seq_txn[-1]["time"])
            train_views  = _filter_by_cutoff(views, cutoff_train)
            train_reviews= _filter_by_cutoff(revs,  cutoff_train)

            train_set.append((
                user_id,
                {
                    "transactions": train_seq_txn,
                    "views": train_views,
                    "reviews": train_reviews,
                }
            ))
            
            for txn in train_seq_txn:
                if "cat" in txn:
                    cat_counter[txn["cat"]] += 1


        # ===== 2) 验证集：与原逻辑一致——单样本，transactions=train+val =====
        for i in range(len(val_seq_txn)):
            traj_val = train_seq_txn + val_seq_txn[:i+1]
            cutoff_val = parse_ts(traj_val[-1]["time"])
            val_views  = _filter_by_cutoff(views, cutoff_val)
            val_reviews= _filter_by_cutoff(revs,  cutoff_val)

            val_set.append((
                user_id,
                {
                    "transactions": traj_val,
                    "views": val_views,
                    "reviews": val_reviews,
                }
            ))
        # traj_val = train_seq_txn + val_seq_txn
        # cutoff_val = parse_ts(traj_val[-1]["time"])
        # val_views  = _filter_by_cutoff(views, cutoff_val)
        # val_reviews= _filter_by_cutoff(revs,  cutoff_val)

        # val_set.append((
        #     user_id,
        #     {
        #         "transactions": traj_val,
        #         "views": val_views,
        #         "reviews": val_reviews,
        #     }
        # ))

        # ===== 3) 测试集：逐步增长，transactions=train+val+test[:i+1] =====
        for i in range(len(test_seq_txn)):
            traj_test = train_seq_txn + val_seq_txn + test_seq_txn[:i+1]
            cutoff_test = parse_ts(traj_test[-1]["time"])
            test_views   = _filter_by_cutoff(views, cutoff_test)
            test_reviews = _filter_by_cutoff(revs,  cutoff_test)

            test_set.append((
                user_id,
                {
                    "transactions": traj_test,
                    "views": test_views,
                    "reviews": test_reviews,
                }
            ))

    # ## re-index users
    # user2id = {}
    # def _collect(uid):
    #     if uid not in user2id:
    #         user2id[uid] = len(user2id)

    # for uid, _ in train_set: _collect(uid)
    # for uid, _ in val_set:   _collect(uid)
    # for uid, _ in test_set:  _collect(uid)

    # def _remap(dataset, mapping):
    #     return [ (mapping[uid], pack) for uid, pack in dataset ]

    # train_set = _remap(train_set, user2id)
    # val_set   = _remap(val_set,   user2id)
    # test_set  = _remap(test_set,  user2id)
    
    cat_freq = dict(cat_counter)
    print(f"train samples: {len(train_set)}, val samples: {len(val_set)}, test samples: {len(test_set)}")
    
    return train_set, val_set, test_set, cat_freq


def process_dataset(user_trajs, train_ratio, val_ratio):
    """
    Split user trajectories into train/val/test sets.
    For validation and test, generate samples with history growing
    back to the beginning of the training set (no fixed window size).

    Args:
        user_trajs: list of [user_id, user_visits], where user_visits is a list of dicts
        train_ratio: float, proportion of training data
        val_ratio: float, proportion of validation data

    Returns:
        train_set, val_set, test_set
        - train_set: list of (user_id, train_sequence)
        - val_set:   list of (user_id, history, target)
        - test_set:  list of (user_id, history, target)
    """
    train_set, val_set, test_set = [], [], []

    for user_id, visits in user_trajs:
        n = len(visits)
        if n < 5:  # too short to split
            continue

        # split indices
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        # adjust to guarantee at least 1 val and 1 test
        if train_end >= n - 2:  
            train_end = n - 2
        if val_end <= train_end:  
            val_end = train_end + 1
        if val_end >= n:  
            val_end = n - 1


        train_seq = visits[:train_end]
        val_seq   = visits[train_end:val_end]
        test_seq  = visits[val_end:]

        # training: keep as whole sequence
        if len(train_seq) > 1:
            train_set.append((user_id, train_seq))

        # validation: history starts from training sequence
        # for i in range(len(val_seq)):
        #     # history = train_seq + val_seq[:i]
        #     # target = val_seq[i]
        #     traj = train_seq + val_seq[:i+1]
        #     val_set.append((user_id, traj))
        
        if len(val_seq) >= 1:
            val_set.append((user_id, train_seq + val_seq))
            # print(f"{len(train_seq+val_seq)}")
        else:
            print(n, len(train_seq), len(val_seq), len(test_seq))

        # test: history starts from training + validation
        for i in range(len(test_seq)):
            # history = train_seq + val_seq + test_seq[:i]
            # target = test_seq[i]
            traj = train_seq + val_seq + test_seq[:i+1]
            test_set.append((user_id, traj))

    print(f"train samples: {len(train_set)}, val samples: {len(val_set)}, test samples: {len(test_set)}")
    return train_set, val_set, test_set



def compute_loss(user_emb, item_emb):
    # Dot product between all users and items
    logits = torch.matmul(user_emb, item_emb.t())  # [B, B]
    labels = torch.arange(logits.size(0)).to(logits.device)  # 正确的匹配是对角线
    loss = F.cross_entropy(logits, labels)
    return loss



def time_to_index(timestr):
    dt = datetime.strptime(timestr, "%Y-%m-%d %H:%M:%S")
    hour = dt.hour
    weekday = dt.weekday()  # Monday=0, Sunday=6
    if weekday < 5:
        return hour  # Weekday: index 0~23
    else:
        return hour + 24  # Weekend: index 24~47


def time_to_index1(timestr):
    """
    Convert timestamp string to discrete indices:
    - hour of day: 0~23
    - day of week: 0~6 (Monday=0, Sunday=6)
    - minute of hour: 0~59
    """
    dt = datetime.fromisoformat(timestr.replace("Z", "+00:00"))
    # dt = datetime.strptime(timestr, "%Y-%m-%d %H:%M:%S")
    hour = dt.hour
    weekday = dt.weekday()   # Monday=0, Sunday=6
    minute = dt.minute
    return weekday, hour, minute

def move_to_device(batch, device):
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]
    elif isinstance(batch, tuple):
        return tuple(move_to_device(v, device) for v in batch)
    elif hasattr(batch, 'to'):
        return batch.to(device)
    else:
        return batch



def sample_nearby_coord(lat, lon, std_meters=1000):
    # 地球半径（米）
    R = 6378137

    # 在局部近似下，1 纬度 ≈ 111320 米，1 经度 ≈ 111320 * cos(lat)
    dx, dy = np.random.normal(0, std_meters, 2)  # 高斯分布采样

    dlat = dy / 111320  # 米 -> 纬度
    dlon = dx / (111320 * np.cos(np.radians(lat)))  # 米 -> 经度

    return (lat + dlat, lon + dlon)

def check_model(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}")
            
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    
    

def compute_hit_ndcg(scores, target_idx, ks=[1, 5, 10, 20]):
    results = {f"hit@{k}": 0.0 for k in ks}
    results.update({f"ndcg@{k}": 0.0 for k in ks})
    
    B = scores.size(0)  # 当前 batch 中样本数
    
    
    for k in ks:
        _, topk = scores.topk(k, dim=1)  # [B, k]
        target = target_idx.unsqueeze(1)  # [B, 1]
        hit = (topk == target)  # [B, k], 0 or 1
        
        sample_hit = hit.any(dim=1).float()  # 0 or 1 for each sample
        hit_k = sample_hit.sum().item()      # 命中的样本数（不是总命中次数）
        results[f"hit@{k}"] = hit_k

        # nDCG@k: 计算真实目标排名的打分
        ranks = hit.nonzero(as_tuple=False)  # [?, 2]，列0是 batch idx，列1是 rank
        ndcg = torch.zeros(B, device=scores.device)

        if ranks.numel() > 0:
            ndcg[ranks[:, 0]] = 1.0 / torch.log2((ranks[:, 1] + 2).float())

        results[f"ndcg@{k}"] = ndcg.sum().item()
        
    print(f"[DEBUG] Batch size = {B}, Hit@20 count = {results['hit@20']:.2f}")

    return results

def ndcg_k(topk_results, k):
    # 归一化折损累计增益（NDCG）
    ndcg = 0.0
    for row in topk_results:
        res = row[:k]
        one_ndcg = 0.0
        for i in range(len(res)):
            one_ndcg += res[i] / math.log(i + 2, 2)
        ndcg += one_ndcg
    return ndcg


def hit_k(topk_results, k):
    hit = 0.0
    for row in topk_results:
        res = row[:k]
        if sum(res) > 0:
            hit += 1
    return hit


def ndcg_k_one(result, k):
    # 归一化折损累计增益（NDCG）
    ndcg = 0.0
    res = result[:k]
    one_ndcg = 0.0
    for i in range(len(res)):
        one_ndcg += res[i] / math.log(i + 2, 2)
    ndcg += one_ndcg
    return ndcg


def hit_k_one(result, k):
    hit = 0.0
    res = result[:k]
    if sum(res) > 0:
        hit += 1
    return hit


@torch.no_grad()
def get_all_poi_embedding(model, poi_dict, poi_cont_normalizers, device):
    N = len(poi_dict)
    model.eval()
    # 初始化特征矩阵，包含 idx=0 的 padding 行
    poi_cat = torch.zeros(N + 1, dtype=torch.long, device=device)       # [N+1]
    poi_cont = torch.zeros(N + 1, 1, dtype=torch.float, device=device)  # [N+1, 1]
    poi_coords = torch.zeros(N + 1, 2, dtype=torch.float, device=device)  # [N+1, 2]
    poi_idxs = torch.arange(N + 1, dtype=torch.long, device=device)       # [0, 1, ..., N]

    for p in poi_dict.values():
        idx = p['idx']
        rating_norm = poi_cont_normalizers['rating'].transform([[p['rating']]])[0][0]
        lat_norm = poi_cont_normalizers['lat'].transform([[p['lat']]])[0][0]
        lon_norm = poi_cont_normalizers['lon'].transform([[p['lon']]])[0][0]
        poi_cat[idx] = p['cat']
        poi_cont[idx] = rating_norm
        poi_coords[idx] = torch.tensor([lat_norm, lon_norm], device=device)

    poi_cat = {'cat': poi_cat}
    poi_emb = model.get_poi_embedding(poi_idxs, poi_cat, poi_cont, poi_coords)  # [N+1, D]
    return poi_emb





@torch.no_grad()
def evaluate_nxt(model, eval_loader, device, idx2cat, k_list=[1, 5, 10, 20], mode='train'):
    """
    eval_loader: 用户样本，每个包含 target_poi 的 ground truth
    """
    model.eval()
    num_samples = 0
    hit_res = [0]*len(k_list)
    ndcg_res = [0]*len(k_list)
    cat_hit_res = [0]*len(k_list)
    
    # N = all_poi_emb.shape[0]

    for batch in eval_loader:
        batch = move_to_device(batch, device)
        user_cat, user_cont = batch['user_cat'], batch['user_cont']
        poi_idxs, target_time, target_loc, target_cat = batch['target_poi'], batch['target_time'], batch['target_coords'], batch['target_cat']
        # history, history_lengths = batch['history'], batch['history_length']
            
        trans_history  = batch['history']         
        view_history   = batch['view_history']
        review_history = batch['review_history']
        
        
        poi_table = model.poi_reps.weight
        user_emb, cat_pred_logits = model.user_tower(user_cat, user_cont, trans_history, view_history, review_history, target_time, target_loc)
        
      
        cat_topk_preds = torch.topk(cat_pred_logits, k=k_list[-1], dim=1).indices  # shape: [B, max_k]
        cats_print = cat_topk_preds.int().cpu().tolist()
        
        if mode is not 'train':
            for cats in cats_print:
                cats_name = [idx2cat[idx] for idx in cats]
                print(f"{cats_name[:5]}")
                
        cat_one_results = (target_cat.reshape(-1,1) == cat_topk_preds).int().cpu().tolist()
        
        
        scores = torch.matmul(user_emb, poi_table.t())  ## [B, num_pois]


        top_k_preds = torch.topk(scores, k=k_list[-1], dim=1).indices   #shape: (batch_size, k)
        one_results = (poi_idxs.reshape(-1,1) == top_k_preds).int().cpu().tolist()

        num_samples += len(poi_idxs)

        for i, k in enumerate(k_list):
            hit_k_batch = hit_k(one_results, k)
            ndcg_k_batch = ndcg_k(one_results, k)
            cat_k_batch = hit_k(cat_one_results, k)
            hit_res[i] += hit_k_batch
            ndcg_res[i] += ndcg_k_batch
            cat_hit_res[i] += cat_k_batch
        
        
    ndcg_res = [ndcg/num_samples for ndcg in ndcg_res]
    hit_res = [hit/num_samples for hit in hit_res]
    cat_hit_res = [x / num_samples for x in cat_hit_res]


    print(f'Test results: total num of test samples {num_samples}\n')
    print(f'hit@1: {hit_res[0]:.4f}, hit@5: {hit_res[1]:.4f}, hit@10: {hit_res[2]:.4f}, hit@20: {hit_res[3]:.4f}\n')
    print(f'category hit@1: {cat_hit_res[0]:.4f}, hit@5: {cat_hit_res[1]:.4f}, hit@10: {cat_hit_res[2]:.4f}, hit@20: {cat_hit_res[3]:.4f}\n')
    print(f'ndcg@1: {ndcg_res[0]:.4f}, ndcg@5: {ndcg_res[1]:.4f}, ndcg@10: {ndcg_res[2]:.4f}, ndcg@20: {ndcg_res[3]:.4f}, \n')
    print(np.sum(hit_res))
    
    return np.sum(hit_res)


@torch.no_grad()
def evaluate_nxt_cat(model, eval_loader, device, idx2cat, k_list=[1, 5, 10, 20], mode='train'):
    """
    eval_loader: 用户样本，每个包含 target_poi 的 ground truth
    """
    model.eval()
    num_samples = 0
    hit_res = [0]*len(k_list)
    ndcg_res = [0]*len(k_list)
    cat_hit_res = [0]*len(k_list)
    
    # N = all_poi_emb.shape[0]

    for batch in eval_loader:
        batch = move_to_device(batch, device)
        user_cat, user_cont = batch['user_cat'], batch['user_cont']
        poi_idxs, target_time, target_loc, target_cat = batch['target_poi'], batch['target_time'], batch['target_coords'], batch['target_cat']
        # history, history_lengths = batch['history'], batch['history_length']
            
        trans_history  = batch['history']         
        view_history   = batch['view_history']
        review_history = batch['review_history']
        
    
        user_emb, cat_pred_logits = model.user_tower(user_cat, user_cont, trans_history, view_history, review_history, target_time, target_loc)
        
      
        cat_topk_preds = torch.topk(cat_pred_logits, k=k_list[-1], dim=1).indices  # shape: [B, max_k]
        cats_print = cat_topk_preds.int().cpu().tolist()
        
        if mode is not 'train':
            for cats in cats_print:
                cats_name = [idx2cat[idx] for idx in cats]
                print(f"{cats_name[:5]}")
       
        cat_one_results = (target_cat.reshape(-1,1) == cat_topk_preds).int().cpu().tolist()
        
        
        # scores = torch.matmul(user_emb, poi_table.t())  ## [B, num_pois]


        # top_k_preds = torch.topk(scores, k=k_list[-1], dim=1).indices   #shape: (batch_size, k)
        # one_results = (poi_idxs.reshape(-1,1) == top_k_preds).int().cpu().tolist()

        num_samples += len(poi_idxs)

        for i, k in enumerate(k_list):
            # hit_k_batch = hit_k(one_results, k)
            # ndcg_k_batch = ndcg_k(one_results, k)
            cat_k_batch = hit_k(cat_one_results, k)
            # hit_res[i] += hit_k_batch
            # ndcg_res[i] += ndcg_k_batch
            cat_hit_res[i] += cat_k_batch
        
        
    # ndcg_res = [ndcg/num_samples for ndcg in ndcg_res]
    # hit_res = [hit/num_samples for hit in hit_res]
    cat_hit_res = [x / num_samples for x in cat_hit_res]


    print(f'Test results: total num of test samples {num_samples}\n')
    # print(f'hit@1: {hit_res[0]:.4f}, hit@5: {hit_res[1]:.4f}, hit@10: {hit_res[2]:.4f}, hit@20: {hit_res[3]:.4f}\n')
    print(f'category hit@1: {cat_hit_res[0]:.4f}, hit@5: {cat_hit_res[1]:.4f}, hit@10: {cat_hit_res[2]:.4f}, hit@20: {cat_hit_res[3]:.4f}\n')
    # print(f'ndcg@1: {ndcg_res[0]:.4f}, ndcg@5: {ndcg_res[1]:.4f}, ndcg@10: {ndcg_res[2]:.4f}, ndcg@20: {ndcg_res[3]:.4f}, \n')
    print(np.sum(cat_hit_res))
    
    return np.sum(cat_hit_res)

@torch.no_grad()
def evaluate(model, eval_loader, all_poi_emb, device, k_list=[1, 5, 10, 20]):
    """
    eval_loader: 用户样本，每个包含 target_poi 的 ground truth
    """
    model.eval()
    num_samples = 0
    hit_res = [0]*len(k_list)
    ndcg_res = [0]*len(k_list)
    cat_hit_res = [0]*len(k_list)
    
    N = all_poi_emb.shape[0]

    for batch in eval_loader:
        batch = move_to_device(batch, device)
        user_cat, user_cont = batch['user_cat'], batch['user_cont']
        poi_idxs, target_time, target_loc, target_cat = batch['target_poi'], batch['target_time'], batch['target_coords'], batch['target_cat']
        history, history_lengths = batch['history_pois'], batch['history_length']
        
        

        user_emb, cat_logits = model.get_user_embedding(user_cat, user_cont, history, target_time, target_loc)
        
        # poi_cat, poi_cont, poi_coords = batch['poi_cat'], batch['poi_cont'], batch['poi_coords']
        # user_emb1, poi_emb = model(user_cat, user_cont, history, target_time, target_loc,
                                        # poi_idxs, poi_cat, poi_cont, poi_coords)
                                        
        cat_topk_preds = torch.topk(cat_logits, k=k_list[-1], dim=1).indices  # shape: [B, max_k]
        cat_one_results = (target_cat.reshape(-1,1) == cat_topk_preds).int().cpu().tolist()
        
        
        scores = torch.matmul(user_emb, all_poi_emb.t())  ## [B, num_pois]
        
        # print(f"scores.shape: {scores.shape}")
        # print(f"poi_idxs[:10]: {poi_idxs[:10]}")
        # print(f"Top-10 indices of sample :10 : {[score.topk(10).indices for score in scores[:10]]}")
        
        # print(f"poi emb comparison {all_poi_emb[poi_idxs[0]]}, \n {poi_emb[0]}")
        # print(f"user emb comparison {user_emb[0]}, \n {user_emb1[0]}")


        top_k_preds = torch.topk(scores, k=k_list[-1], dim=1).indices   #shape: (batch_size, k)
        one_results = (poi_idxs.reshape(-1,1) == top_k_preds).int().cpu().tolist()

        num_samples += len(poi_idxs)

        for i, k in enumerate(k_list):
            hit_k_batch = hit_k(one_results, k)
            ndcg_k_batch = ndcg_k(one_results, k)
            cat_k_batch = hit_k(cat_one_results, k)
            hit_res[i] += hit_k_batch
            ndcg_res[i] += ndcg_k_batch
            cat_hit_res[i] += cat_k_batch
        
        
    ndcg_res = [ndcg/num_samples for ndcg in ndcg_res]
    hit_res = [hit/num_samples for hit in hit_res]
    cat_hit_res = [x / num_samples for x in cat_hit_res]


    print(f'Test results: total num of test samples {num_samples}\n')
    print(f'hit@1: {hit_res[0]:.4f}, hit@5: {hit_res[1]:.4f}, hit@10: {hit_res[2]:.4f}, hit@20: {hit_res[3]:.4f}\n')
    print(f'category hit@1: {cat_hit_res[0]:.4f}, hit@5: {cat_hit_res[1]:.4f}, hit@10: {cat_hit_res[2]:.4f}, hit@20: {cat_hit_res[3]:.4f}\n')
    print(f'ndcg@1: {ndcg_res[0]:.4f}, ndcg@5: {ndcg_res[1]:.4f}, ndcg@10: {ndcg_res[2]:.4f}, ndcg@20: {ndcg_res[3]:.4f}, \n')
    
    return np.sum(hit_res)




def get_history_group(length):
    if length < 10:
        return "<10"
    elif length < 20:
        return "10-20"
    elif length < 40:
        return "20-40"
    elif length < 60:
        return "40-60"
    elif length < 80:
        return "60-80"
    elif length < 100:
        return "80-100"
    else:
        return ">100"
    

@torch.no_grad()
def evaluate_length_bucket(model, eval_loader, all_poi_emb, device, k_list=[1, 5, 10, 20]):
    """
    eval_loader: 用户样本，每个包含 target_poi 的 ground truth
    """
    model.eval()
    num_samples = 0
    # hit_res = [0]*len(k_list)
    # ndcg_res = [0]*len(k_list)
    
    hit_res = defaultdict(lambda: [0] * len(k_list))
    ndcg_res = defaultdict(lambda: [0] * len(k_list))
    count_res = defaultdict(int)
    
    N = all_poi_emb.shape[0]

    for batch in eval_loader:
        batch = move_to_device(batch, device)
        user_cat, user_cont = batch['user_cat'], batch['user_cont']
        poi_idxs, target_time, target_loc = batch['target_poi'], batch['target_time'], batch['target_coords']
        history, history_lengths = batch['history_pois'], batch['history_length']
        
        

        user_emb = model.get_user_embedding(user_cat, user_cont, history, target_time, target_loc)
        
        scores = torch.matmul(user_emb, all_poi_emb.t())  ## [B, num_pois]


        top_k_preds = torch.topk(scores, k=k_list[-1], dim=1).indices   #shape: (batch_size, k)
        one_results = (poi_idxs.reshape(-1,1) == top_k_preds).int().cpu().tolist()

        num_samples += len(poi_idxs)

        # for i, k in enumerate(k_list):
        #     hit_k_batch = hit_k(one_results, k)
        #     ndcg_k_batch = ndcg_k(one_results, k)
        #     hit_res[i] += hit_k_batch
        #     ndcg_res[i] += ndcg_k_batch
        
        for idx, result in enumerate(one_results):
            h_len = history_lengths[idx]
            group = get_history_group(h_len)
            count_res[group] += 1
            for i, k in enumerate(k_list):
                hit_res[group][i] += hit_k_one(result, k)
                ndcg_res[group][i] += ndcg_k_one(result, k)
        
        
    # ndcg_res = [ndcg/num_samples for ndcg in ndcg_res]
    # hit_res = [hit/num_samples for hit in hit_res]

    # print(f'Test results: total num of test samples {num_samples}\n')
    # print(f'hit@1: {hit_res[0]:.4f}, hit@5: {hit_res[1]:.4f}, hit@10: {hit_res[2]:.4f}, hit@20: {hit_res[3]:.4f}\n')
    # print(f'ndcg@1: {ndcg_res[0]:.4f}, ndcg@5: {ndcg_res[1]:.4f}, ndcg@10: {ndcg_res[2]:.4f}, ndcg@20: {ndcg_res[3]:.4f}, \n')
    
    for group in ['<10', '10-20', '20-40', '40-60', '60-80', '80-100', '>100']:
        if count_res[group] == 0:
            continue
        print(f"\nHistory Length Group: {group} (count={count_res[group]})")
        for i, k in enumerate(k_list):
            avg_hit = hit_res[group][i] / count_res[group]
            avg_ndcg = ndcg_res[group][i] / count_res[group]
            print(f"  Hit@{k}: {avg_hit:.4f}, NDCG@{k}: {avg_ndcg:.4f}")
    
    return np.sum(hit_res['>100'])



def cosine_with_warmup_schedule(steps_per_epoch, num_epochs, warmup_epochs):
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = steps_per_epoch * warmup_epochs

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


def print_result(metrics):
    print(f"Hit@1: {metrics['hit@1']:.4f}, Hit@5: {metrics['hit@5']:.4f}, Hit@10: {metrics['hit@10']:.4f}, Hit@20: {metrics['hit@20']:.4f}")
    print(f"NDCG@5: {metrics['ndcg@5']:.4f}, NDCG@10: {metrics['ndcg@10']:.4f}, NDCG@20: {metrics['ndcg@20']:.4f}")




if __name__ == '__main__':
    user_data, poi_dict, user_dict = read_dataset("../data/SG_synthetic/app_profiles_all_users_version_4.json",
                                                  "../data/SG_synthetic/all_poi_id_all_interactions.json")
    
    user_cat_dims, user_cont_normalizers = process_user_info(user_dict)
    poi_cat_dims, poi_cont_normalizers = process_poi_info(poi_dict)
    num_pois = len(poi_dict) + 1 ## +1 for padding 
    print(user_cat_dims, user_cont_normalizers)
    print(num_pois, poi_cat_dims, poi_cont_normalizers)