import torch
from torch.utils.data import Dataset
from collections import defaultdict
from utils import time_to_index1, sample_nearby_coord, parse_ts



def build_side(cutoff_dt, seq, max_len, poi_dict):
    # seq includes: {'poi','cat','time'}
    filt = [e for e in seq if e.get('time') and parse_ts(e['time']) <= cutoff_dt]
    # cutoff num: max_len
    if len(filt) > max_len:
        filt = filt[-max_len:]
    pois_idx = [poi_dict[e['poi']]['idx'] for e in filt]
    cats_idx = [e['cat'] for e in filt]
    times_enc = [time_to_index1(e['time']) for e in filt]
    return pois_idx, cats_idx, times_enc


class TrainDataset(Dataset):
    def __init__(self, records, user_dict, user_cont_normalizers, poi_dict, poi_cont_normalizers, max_history_len=100):
        """
        Args:
            records (list of dict): List of user behavior records. Each record must have 'user_id', 'item_id', and 'timestamp'.
            split_mode (str): 'train', 'val', or 'test'.
            max_history_len (int): Maximum history length to keep.
        """
        self.max_history_len = max_history_len
        self.user_dict = user_dict
        self.user_cont_normalizers = user_cont_normalizers
        self.poi_dict = poi_dict
        self.poi_cont_normalizers = poi_cont_normalizers

        self.user_histories = records
        
        self.samples = []


        for (user_id, pack) in self.user_histories:
            # print(user_id, pack)
            txns   = pack.get("transactions", []) or []
            views  = pack.get("views", []) or []
            reviews= pack.get("reviews", []) or []

            if len(txns) < 2:
                # Not enough interactions for train/val/test, optionally skip or handle separately
                continue
            txn_items = [e['poi'] for e in txns]
            txn_cats  = [e['cat'] for e in txns]
            txn_times_raw = [e['time'] for e in txns]
            txn_times = [time_to_index1(t) for t in txn_times_raw]
            
            n = len(txns)
            for i in range(1, n):
                pois = txn_items[:i]
                cats = txn_cats[:i]
                times = txn_times[:i]

                target = txn_items[i]
                target_time = txn_times[i]
                target_cat = txn_cats[i]
                
                target_loc = sample_nearby_coord(poi_dict[txn_items[i]]['lat'], poi_dict[txn_items[i]]['lon'])
                
                if len(pois) > self.max_history_len:
                    pois  = pois[-self.max_history_len:]
                    cats  = cats[-self.max_history_len:]
                    times = times[-self.max_history_len:]
                
                
                cutoff_dt = parse_ts(txn_times_raw[i-1])
                view_pois_idx, view_cats, view_times = build_side(cutoff_dt, views, self.max_history_len, poi_dict)
                review_pois_idx, review_cats, review_times = build_side(cutoff_dt, reviews, self.max_history_len, poi_dict)

                self.samples.append({
                    'user_id': user_id,
                    ## main seq: transaction history
                    'pois': pois,
                    'times': times,
                    'cats': cats,
                    ## side seq: 
                    'view_history': {
                        'pois': view_pois_idx,
                        'history_cats': view_cats,
                        'history_times': view_times,
                    },
                    'review_history': {
                        'pois': review_pois_idx,
                        'history_cats': review_cats,
                        'history_times': review_times,
                    },
                    'target': target,
                    'target_time': target_time,
                    'target_loc': target_loc,
                    'target_cat': target_cat,
                })


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        user_id = sample['user_id']
        # history = sample['pois']
        history = [self.poi_dict[poi_id]['idx'] for poi_id in sample['pois']]
        history_cats = sample['cats']
        history_times = sample['times']
        
        target = sample['target']
        target_time = sample['target_time']
        target_loc = sample['target_loc']
        
        poi_feat = self.poi_dict[target]
        # poi_idx = self.poi_dict[target]['poi']
        user_feat = self.user_dict[user_id]
        
        age_norm = self.user_cont_normalizers['age'].transform([[user_feat['age']]])[0][0]
        rating_norm = self.poi_cont_normalizers['rating'].transform([[poi_feat['rating']]])[0][0]
        lat_norm = self.poi_cont_normalizers['lat'].transform([[poi_feat['lat']]])[0][0]
        lon_norm = self.poi_cont_normalizers['lon'].transform([[poi_feat['lon']]])[0][0]
        
        target_lat_norm = self.poi_cont_normalizers['lat'].transform([[target_loc[0]]])[0][0]
        target_lon_norm = self.poi_cont_normalizers['lon'].transform([[target_loc[1]]])[0][0]

        return {
            'history': {
                'pois':history,
                'history_cats': history_cats,
                'history_times': history_times},
            'view_history': sample['view_history'],
            'review_history': sample['review_history'],
            'user_cat': {
                'gender': user_feat['gender'],
                'device': user_feat['device']
            },
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'user_cont': {
                'age': age_norm
            },
            
            'poi_cat': {
                'cat': poi_feat['cat']
            },
            'poi_cont': {
                'rating': rating_norm,
                'lat': lat_norm,
                'lon': lon_norm,
            },
            
            'target':{
                'target_poi': poi_feat['idx'],
                'target_lat': target_lat_norm,
                'target_lon': target_lon_norm,
                'target_time': target_time,
                'target_cat': sample['target_cat']
            },
            
            'length': torch.tensor(len(history), dtype=torch.long),
            'length_view': torch.tensor(len(sample['view_history']['pois']), dtype=torch.long),
            'length_review': torch.tensor(len(sample['review_history']['pois']), dtype=torch.long),
        }



class TestDataset(Dataset):
    def __init__(self, records, user_dict, user_cont_normalizers, poi_dict, poi_cont_normalizers, max_history_len=50):
        """
        Args:
            records (list of dict): List of user behavior records. Each record must have 'user_id', 'item_id', and 'timestamp'.
            split_mode (str): 'train', 'val', or 'test'.
            max_history_len (int): Maximum history length to keep.
        """
        self.max_history_len = max_history_len
        self.user_dict = user_dict
        self.user_cont_normalizers = user_cont_normalizers
        self.poi_dict = poi_dict
        self.poi_cont_normalizers = poi_cont_normalizers

        self.user_histories = records
        
        self.samples = []
        # print(len(self.user_histories))
        for (user_id, pack) in self.user_histories:
            txn_seq   = pack.get("transactions", []) or []
            view_seq  = pack.get("views", []) or []
            review_seq= pack.get("reviews", []) or []
            
            
            n = len(txn_seq)
            
            history_txn = txn_seq[:-1]
            target_txn  = txn_seq[-1]
            cutoff_dt = parse_ts(history_txn[-1]["time"])
            
            hist_pois  = [e["poi"] for e in history_txn]
            hist_cats  = [e["cat"] for e in history_txn]
            hist_times = [time_to_index1(e["time"]) for e in history_txn]

            target = target_txn["poi"]
            target_time = time_to_index1(target_txn["time"])
            target_cat = target_txn["cat"]
            target_loc = sample_nearby_coord(target_txn['lat'], target_txn['lon'])
            

            if len(hist_pois) > self.max_history_len:
                hist_pois  = hist_pois[-self.max_history_len:]
                hist_cats  = hist_cats[-self.max_history_len:]
                hist_times = hist_times[-self.max_history_len:]
            
            
            view_pois_idx, view_cats, view_times = build_side(cutoff_dt, view_seq, self.max_history_len, poi_dict)
            review_poi_idx, review_cats, review_times = build_side(cutoff_dt, review_seq, self.max_history_len, poi_dict)
            
            self.samples.append({
                "user_id": user_id,
                "hist_pois": hist_pois,         
                "hist_cats": hist_cats,
                "hist_times": hist_times,
                "view_history": {
                    "pois": view_pois_idx,      
                    "history_cats": view_cats,
                    "history_times": view_times,
                },
                "review_history": {
                    "pois": review_poi_idx,     
                    "history_cats": review_cats,
                    "history_times": review_times,
                },
                "target_poi": target,
                "target_cat": target_cat,
                "target_time": target_time,
                'target_loc': target_loc,
            })
            
                
            
                


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        user_id = sample['user_id']
        history = [self.poi_dict[poi_id]['idx'] for poi_id in sample['hist_pois']]
        history_cats = sample['hist_cats']
        history_times = sample['hist_times']
        
        target = sample['target_poi']
        target_loc = sample['target_loc']
        target_time = sample['target_time']
        
        poi_feat = self.poi_dict[target]
        user_feat = self.user_dict[user_id]
        # poi_idx = self.poi_dict[target]['idx']
        
        age_norm = self.user_cont_normalizers['age'].transform([[user_feat['age']]])[0][0]
        rating_norm = self.poi_cont_normalizers['rating'].transform([[poi_feat['rating']]])[0][0]
        lat_norm = self.poi_cont_normalizers['lat'].transform([[poi_feat['lat']]])[0][0]
        lon_norm = self.poi_cont_normalizers['lon'].transform([[poi_feat['lon']]])[0][0]
        
        target_lat_norm = self.poi_cont_normalizers['lat'].transform([[target_loc[0]]])[0][0]
        target_lon_norm = self.poi_cont_normalizers['lon'].transform([[target_loc[1]]])[0][0]
        

        
        return {
            'history': {
                'pois':history,
                'history_cats': history_cats,
                'history_times': history_times},
            "view_history": sample["view_history"],
            "review_history": sample["review_history"],
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'user_cat': {
                'gender': user_feat['gender'],
                'device': user_feat['device']
            },
            'user_cont': {
                'age': age_norm
            },
            'poi_cat': {
                'cat': poi_feat['cat']
            },
            'poi_cont': {
                'rating': rating_norm,
                'lat': lat_norm,
                'lon': lon_norm,
            },
            'target':{
                'target_poi': poi_feat['idx'], 
                'target_lat': target_lat_norm,
                'target_lon': target_lon_norm,
                'target_time': sample['target_time'],
                'target_cat': sample['target_cat']
            },
            'length': torch.tensor(len(history), dtype=torch.long),
            "length_view": torch.tensor(len(sample["view_history"]["pois"]), dtype=torch.long),
            "length_review": torch.tensor(len(sample["review_history"]["pois"]), dtype=torch.long),
        }



def collate_fn(batch):
    """
    Collate function to pad variable-length history sequences on the left side.
    Args:
        batch: List of samples, each is a dict with keys 'history_items', 'target_item', 'history_length'
    Returns:
        batch_history_items: (batch_size, max_seq_len) LongTensor
        batch_history_mask: (batch_size, max_seq_len) BoolTensor
        batch_target_items: (batch_size,) LongTensor
        batch_history_lengths: (batch_size,) LongTensor
    """
    # 1. Left Padding History
    pois = [torch.tensor(sample['history']['pois'],dtype=torch.long) for sample in batch]
    # times = [torch.tensor(sample['times'],dtype=torch.long) for sample in batch]
    cats = [torch.tensor(sample['history']['history_cats'],dtype=torch.long) for sample in batch]
    times = [torch.tensor(sample['history']['history_times'],dtype=torch.long) for sample in batch]
    
    target_pois = torch.tensor([sample['target']['target_poi'] for sample in batch], dtype=torch.long).unsqueeze(-1)
    target_poi_coords = torch.tensor([[x['target']['target_lat'], x['target']['target_lon']] for x in batch], dtype=torch.float)
    target_time = torch.tensor([x['target']['target_time'] for x in batch], dtype=torch.long)
    target_cat = torch.tensor([x['target']['target_cat'] for x in batch], dtype=torch.long)
    
    history_lengths = torch.stack([sample['length'] for sample in batch])

    # Find maximum length in the batch
    max_len = max([len(seq) for seq in pois])

    # Do left padding manually
    padded_histories, padded_cats, padded_times = [], [], []
    attention_masks = []
    for (poi, cat, time) in zip(pois, cats, times):
        pad_len = max_len - len(poi)
        padded_hist = torch.cat([torch.full((pad_len,), 0, dtype=torch.long), poi])  # padding on the left
        padded_cat = torch.cat([torch.full((pad_len,), 0, dtype=torch.long), cat])
        padded_time = torch.cat([torch.full((pad_len, 3), 0, dtype=torch.long), time])
        mask = torch.cat([torch.ones(pad_len, dtype=torch.bool), torch.zeros(len(poi), dtype=torch.bool)])
        
        padded_histories.append(padded_hist)
        padded_cats.append(padded_cat)
        padded_times.append(padded_time)
        attention_masks.append(mask)

    batch_history_pois = torch.stack(padded_histories)  # (batch_size, max_seq_len)
    batch_attention_mask = torch.stack(attention_masks)  # (batch_size, max_seq_len)
    batch_history_cats = torch.stack(padded_cats)
    batch_history_times = torch.stack(padded_times)
    
    
    
    view_pois = [torch.tensor(s["view_history"]["pois"], dtype=torch.long) for s in batch]
    view_cats = [torch.tensor(s["view_history"]["history_cats"], dtype=torch.long) for s in batch]
    view_times = [torch.tensor(s['view_history']['history_times'],dtype=torch.long) for s in batch]
    
    max_view_len = max([len(seq) for seq in view_pois])
    view_padded_histories, view_padded_cats, view_padded_times = [], [], []
    view_attention_masks = []
    for (poi, cat, time) in zip(view_pois, view_cats, view_times):
        pad_len = max_view_len - len(poi)
        padded_hist = torch.cat([torch.full((pad_len,), 0, dtype=torch.long), poi])  # padding on the left
        padded_cat = torch.cat([torch.full((pad_len,), 0, dtype=torch.long), cat])
        padded_time = torch.cat([torch.full((pad_len, 3), 0, dtype=torch.long), time])
        mask = torch.cat([torch.ones(pad_len, dtype=torch.bool), torch.zeros(len(poi), dtype=torch.bool)])
        
        view_padded_histories.append(padded_hist)
        view_padded_cats.append(padded_cat)
        view_padded_times.append(padded_time)
        view_attention_masks.append(mask)

    view_batch_history_pois = torch.stack(view_padded_histories)  # (batch_size, max_seq_len)
    view_batch_attention_mask = torch.stack(view_attention_masks)  # (batch_size, max_seq_len)
    view_batch_history_cats = torch.stack(view_padded_cats)
    view_batch_history_times = torch.stack(view_padded_times)
    
    
    
    review_pois = [torch.tensor(s["review_history"]["pois"], dtype=torch.long) for s in batch]
    review_cats = [torch.tensor(s["review_history"]["history_cats"], dtype=torch.long) for s in batch]
    review_times = [torch.tensor(s['review_history']['history_times'],dtype=torch.long) for s in batch]
    
    max_review_len = max([len(seq) for seq in review_pois])
    review_padded_histories, review_padded_cats, review_padded_times = [], [], []
    review_attention_masks = []
    for (poi, cat, time) in zip(review_pois, review_cats, review_times):
        pad_len = max_review_len - len(poi)
        padded_hist = torch.cat([torch.full((pad_len,), 0, dtype=torch.long), poi])  # padding on the left
        padded_cat = torch.cat([torch.full((pad_len,), 0, dtype=torch.long), cat])
        padded_time = torch.cat([torch.full((pad_len, 3), 0, dtype=torch.long), time])
        mask = torch.cat([torch.ones(pad_len, dtype=torch.bool), torch.zeros(len(poi), dtype=torch.bool)])
        
        review_padded_histories.append(padded_hist)
        review_padded_cats.append(padded_cat)
        review_padded_times.append(padded_time)
        review_attention_masks.append(mask)

    review_batch_history_pois = torch.stack(review_padded_histories)  # (batch_size, max_seq_len)
    review_batch_attention_mask = torch.stack(review_attention_masks)  # (batch_size, max_seq_len)
    review_batch_history_cats = torch.stack(review_padded_cats)
    review_batch_history_times = torch.stack(review_padded_times)
    
    
    

    
    
    # User categorical features
    user_id = torch.stack([sample['user_id'] for sample in batch])
    user_gender = torch.tensor([x['user_cat']['gender'] for x in batch], dtype=torch.long)
    user_device = torch.tensor([x['user_cat']['device'] for x in batch], dtype=torch.long)

    # User continuous
    user_age = torch.tensor([x['user_cont']['age'] for x in batch], dtype=torch.float).unsqueeze(1)

    # POI categorical
    poi_cat = torch.tensor([x['poi_cat']['cat'] for x in batch], dtype=torch.long)

    # POI continuous
    poi_cont = torch.tensor([
        [x['poi_cont']['rating']]
        for x in batch
    ], dtype=torch.float)
    
    poi_coords = torch.tensor([
        [x['poi_cont']['lat'], x['poi_cont']['lon']] for x in batch], dtype=torch.float)
    
    
    
    
    return {
        'history':{
            'loc_ids': batch_history_pois,      # (batch_size, max_seq_len)
            'loc_mask': batch_attention_mask,      # (batch_size, max_seq_len)
            'loc_cats': batch_history_cats,
            'weekday': batch_history_times[..., 0],
            'hour': batch_history_times[..., 1],
            'minute': batch_history_times[..., 2],
        },
        'view_history':{
            'loc_ids': view_batch_history_pois,      # (batch_size, max_seq_len)
            'loc_mask': view_batch_attention_mask,      # (batch_size, max_seq_len)
            'loc_cats': view_batch_history_cats,
            'weekday': view_batch_history_times[..., 0],
            'hour': view_batch_history_times[..., 1],
            'minute': view_batch_history_times[..., 2],
        },
        'review_history':{
            'loc_ids': review_batch_history_pois,      # (batch_size, max_seq_len)
            'loc_mask': review_batch_attention_mask,      # (batch_size, max_seq_len)
            'loc_cats': review_batch_history_cats,
            'weekday': review_batch_history_times[..., 0],
            'hour': review_batch_history_times[..., 1],
            'minute': review_batch_history_times[..., 2],
        },
        'target_poi': target_pois,               # (batch_size,)
        'target_coords': target_poi_coords,
        'target_time': target_time,
        'target_cat': target_cat,
        'history_length': history_lengths,          # (batch_size,)
        'user_cat': {
            'user_id': user_id,
            'gender': user_gender,
            'device': user_device,
        },
        'user_cont': user_age,           # [batch_size, 1]
        'poi_cat': {
            'cat': poi_cat
        },
        'poi_cont': poi_cont,             # [B, 1]
        'poi_coords': poi_coords         # [B, 2]
        
    }
