import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")


def assert_finite(t, name):
    assert torch.isfinite(t).all(), f"{name} has NaN/Inf; stats: min={t.min().item()} max={t.max().item()}"

# User Tower
class UserTower(nn.Module):
    def __init__(self, poi_reps, user_cat_dims, user_cont_dim, history_len, embed_dim, poi_cat_class, hidden_dim=256, hist_dim=256, dropout=0.1, user_mode='feat'):
        super(UserTower, self).__init__()
        self.user_mode = user_mode
        self.hidden_dim = hidden_dim
        # --- User tower ---
        self.user_embeds = nn.ModuleDict({
            key: nn.Embedding(num_classes, embed_dim)
            for key, num_classes in user_cat_dims.items()
        })
        self.user_cont_proj = nn.Linear(user_cont_dim, embed_dim)
        # self.context_proj = nn.Linear(3, embed_dim)  # (hour, lat, lon)
        
        self.user_id_embeds = nn.Embedding(user_cat_dims['user_id'], hist_dim)
        self.poi_reps = poi_reps
        # self.transformer = HistoryEncoder(hist_dim, max_len=history_len)
        # self.time_emb = nn.Embedding(48, hist_dim)
        
        self.trans_encoder = HistoryEncoder(hist_dim, max_len=history_len)
        self.view_encoder  = HistoryEncoder(hist_dim, max_len=history_len)
        self.review_encoder= HistoryEncoder(hist_dim, max_len=history_len)
        
        
        self.hour_embed = nn.Embedding(24, hidden_dim)
        self.weekday_embed = nn.Embedding(7, hidden_dim)
        self.loc_rep = nn.Linear(2, hist_dim)

        self.cat_embed = nn.Embedding(poi_cat_class, hidden_dim)
        
        self.fusion = BehaviorFusion(dim=hist_dim, ctx_dim=2*hist_dim)

        if user_mode != 'feat':
            self.user_proj = nn.Sequential(
                nn.Linear(embed_dim*(1+len(self.user_embeds))+3*hist_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            )
        
            ## Add category prediction
            self.num_classes = poi_cat_class
            self.category_predictor = nn.Sequential(
                nn.Linear(embed_dim*(1+len(self.user_embeds))+3*hist_dim, 128),
                nn.ReLU(),
                nn.Linear(128, self.num_classes),
            )
        else:
            self.user_proj = nn.Sequential(
            nn.Linear(3*hist_dim, hist_dim),
            nn.ReLU(),
            nn.Linear(hist_dim, hist_dim),
            )
            
            self.fuse_block = FCResidualBlock(dim=hist_dim, hidden=2*hist_dim, dropout=dropout)


            self.category_predictor = nn.Sequential(
                nn.Linear(3*hist_dim, hist_dim),
                nn.ReLU(),
                nn.Linear(hist_dim, poi_cat_class),
            )
        
    
    def build_step_embed(self, loc_ids, loc_cats, hour, weekday):
        
        poi_emb = self.poi_reps(loc_ids) 
        # min_emb = self.minute_embed(minute)
        hour_emb = self.hour_embed(hour)
        day_emb = self.weekday_embed(weekday)
        cxt_emb = self.cat_embed(loc_cats)

        x = poi_emb + hour_emb + day_emb + cxt_emb
        return x  # [B, T, D]
    
    def target_time_emb(self, time):
        # min_emb = self.minute_embed(time[...,2])
        hour_emb = self.hour_embed(time[...,1])
        day_emb = self.weekday_embed(time[...,0])
        
        return hour_emb + day_emb
    
    def encode_history_safe(self, hist: dict, encoder: nn.Module):
        loc_ids = hist['loc_ids']                      # [B, T]
        mask    = (hist['loc_ids'] == 0).bool()  # [B, T], True=pad
        B, T    = loc_ids.shape
        device  = loc_ids.device
        D       = self.hidden_dim  # or hist_dim

        non_empty = (~mask).any(dim=1)                 # [B] bool
        rep = torch.zeros(B, D, device=device)         # 预置输出
        if non_empty.any():
            idx = non_empty.nonzero(as_tuple=False).squeeze(1)      # [B1]
        sub_hist = {k: v.index_select(0, idx) for k, v in hist.items()}  # 每个 [B1, T]

        x = self.build_step_embed(
            loc_ids=sub_hist['loc_ids'],
            loc_cats=sub_hist['loc_cats'],
            hour=sub_hist['hour'],
            weekday=sub_hist['weekday'],
        )                                                         # [B1, T, D]
        enc = encoder(x, mask=sub_hist['loc_mask'])               # [B1, T, D]
        rep_sub = enc[:, -1, :]                                   # 左 pad -> 最后一位是最后一个有效 token（或 pad=0）
        rep.index_copy_(0, idx, rep_sub)                          # 写回

        # 空样本保持 0 向量，避免 NaN
        return rep, non_empty  # rep: [B, D], non_empty: [B]

    def forward(self, user_cat, user_cont, trans_history, view_history, review_history, time, loc):
        B = user_cont.size(0)

        if self.user_mode != 'feat':
            # Categorical
            cat_emb = torch.cat([
                self.user_embeds[k](v) for k, v in user_cat.items()
            ], dim=1)

            # Continuous
            cont_emb = self.user_cont_proj(user_cont)

            # Context (time + location)
            # context_emb = self.context_proj(time_loc)

        # History encoding
        
        trans_mask = (trans_history['loc_ids'] == 0).bool() 
        x_trans = self.build_step_embed(
            loc_ids=trans_history['loc_ids'],
            loc_cats=trans_history['loc_cats'],
            hour=trans_history['hour'],
            weekday=trans_history['weekday'],
        )
        assert_finite(x_trans, "x_trans")
        enc_trans = self.trans_encoder(x_trans, mask=trans_mask)  # [B, T, D]
        assert_finite(enc_trans, "enc_trans")
        rep_trans = enc_trans[:, -1, :]

        
        rep_view,  valid_view    = self.encode_history_safe(view_history,   self.view_encoder)
        rep_review, valid_review = self.encode_history_safe(review_history, self.review_encoder)
        # view_mask = (view_history['loc_ids'] == 0).bool() 
        # x_view = self.build_step_embed(
        #     loc_ids=view_history['loc_ids'],
        #     loc_cats=view_history['loc_cats'],
        #     hour=view_history['hour'],
        #     weekday=view_history['weekday'],
        # )
        # print(x_view.shape, view_history['loc_ids'], view_history['loc_cats'], view_history['weekday'])
        # assert_finite(x_view, "x_view")
        # enc_view = self.view_encoder(x_view, mask=view_mask)      # [B, T, D]
        assert_finite(rep_view, "enc_view")
        # rep_view = enc_view[:, -1, :]

        # review_mask = (review_history['loc_ids'] == 0).bool() 
        # x_review = self.build_step_embed(
        #     loc_ids=review_history['loc_ids'],
        #     loc_cats=review_history['loc_cats'],
        #     hour=review_history['hour'],
        #     weekday=review_history['weekday'],
        # )
        # assert_finite(x_review, "x_review")
        # enc_review = self.review_encoder(x_review, mask=review_mask)  # [B, T, D]
        assert_finite(rep_review, "enc_review")
        # rep_review = enc_review[:, -1, :]
        

        
        ## target context
        # user_feat = torch.cat([cat_emb, cont_emb, context_emb, hist_feat], dim=1)
        time_rep = self.target_time_emb(time)
        loc_rep = self.loc_rep(loc)
        ctx = torch.cat([time_rep, loc_rep], dim=1) # [B, 2D]
        assert_finite(ctx, "ctx")
        
        
        
        ## fusion
        hist_rep, behav_weights = self.fusion([rep_trans, rep_view, rep_review], ctx=ctx)  # [B, D], [B, 3]
        assert_finite(hist_rep, "hist_rep")
        
        
        if self.user_mode != 'feat':
            user_feat = torch.cat([hist_rep, cat_emb, cont_emb, time_rep, loc_rep], dim=1)
        else:
            uid = user_cat['user_id']
            u = self.user_id_embeds(uid)
            fused = self.fuse_block(hist_rep + u)
            user_feat = torch.cat([fused, time_rep, loc_rep], dim=1)
            assert_finite(user_feat, "hist_rep")
            
        
        pred_logits = self.category_predictor(user_feat)

        return self.user_proj(user_feat), pred_logits  # [B, D]


class FCResidualBlock(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.lin1 = nn.Linear(dim, hidden)
        self.lin2 = nn.Linear(hidden, dim)
        self.ln = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = F.relu(self.lin1(self.ln(x)))
        h = self.drop(self.lin2(h))
        return self.ln(x + h)


class HistoryEncoder(nn.Module):
    def __init__(self, embed_dim, max_len=200, num_heads=2, hidden_size=256, num_layers=2, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_size, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, embed_dim))  # max length 200

    def forward(self, x, mask=None):
        # x: [B, T, D]
        B, T, D = x.shape
        pos = self.pos_embedding[:, :T, :].expand(B, -1, -1)
        x = x + pos  # positional encoding
        out = self.encoder(x, src_key_padding_mask=mask)  # [B, T, D]
        return out



class BehaviorFusion(nn.Module):
    """
    行为间注意力融合：把 [trans, view, review] 的最后时刻表示做 softmax 加权和
    支持把 target 的上下文(time/loc)拼上去做门控。
    """
    def __init__(self, dim, ctx_dim=None):
        super().__init__()
        self.use_ctx = ctx_dim is not None and ctx_dim > 0
        in_dim = dim if not self.use_ctx else dim + ctx_dim
        self.scorer = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)  # 对每个行为打分
        )

    def forward(self, reps, ctx=None):
        """
        reps: list of [B, D], 长度=行为数(3)
        ctx:  [B, C] or None
        return: fused [B, D], weights [B, K]
        """
        B, D = reps[0].shape
        K = len(reps)
        X = torch.stack(reps, dim=1)          # [B, K, D]
        if self.use_ctx and ctx is not None:
            ctx_expand = ctx.unsqueeze(1).expand(-1, K, -1)  # [B, K, C]
            X_in = torch.cat([X, ctx_expand], dim=-1)        # [B, K, D+C]
        else:
            X_in = X                                          # [B, K, D]
        scores = self.scorer(X_in).squeeze(-1)                 # [B, K]
        w = torch.softmax(scores, dim=1)                       # [B, K]
        fused = (X * w.unsqueeze(-1)).sum(dim=1)               # [B, D]
        return fused, w


# POI Tower
class POITower(nn.Module):
    def __init__(self,
                 poi_reps, 
                 poi_cat_dims,       # dict: {'poi_type': 50, ...}，
                 poi_cont_dim,
                 embed_dim,
                 rep_dim
                 ):
        super(POITower, self).__init__()
        self.poi_reps = poi_reps
        self.poi_embeds = nn.ModuleDict({
            key: nn.Embedding(num_classes, embed_dim)
            for key, num_classes in poi_cat_dims.items()
        })
        self.poi_cont_proj = nn.Linear(poi_cont_dim, embed_dim)
        self.poi_coord_proj = nn.Linear(2, embed_dim)
        
        self.poi_proj = nn.Sequential(
            nn.Linear(embed_dim *(1+len(poi_cat_dims))+ rep_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
    

    def forward(self, poi_idxs, poi_cat, poi_cont, poi_coords):
        B = poi_cont.size(0)
        
        
        poi_emb = self.poi_reps(poi_idxs).squeeze()

        cat_emb = torch.cat([
            self.poi_embeds[k](v) for k, v in poi_cat.items()
        ], dim=1)
        
        cont_emb = self.poi_cont_proj(poi_cont)
        coord_emb = self.poi_coord_proj(poi_coords)
        # print(cat_emb.shape, poi_emb.shape, cont_emb.shape, coord_emb.shape)
        poi_feat = torch.cat([poi_emb, cat_emb, cont_emb, coord_emb], dim=1)
        
        return self.poi_proj(poi_feat)  # [B, D]


# Combined Model
class RecommendationModel(nn.Module):
    def __init__(self, 
                 num_pois, 
                 user_cat_dims,      # dict: {'user_id': 10000, 'gender': 3, ...}
                 poi_cat_dims,       # dict: {'poi_type': 50, ...}
                 user_cont_dim=2,
                 poi_cont_dim=1,
                 embed_dim=16,
                 hist_dim=256,
                 history_len=200):
        
        super(RecommendationModel, self).__init__()
        self.embed_dim = embed_dim
        self.poi_cat_class = poi_cat_dims['cat']
        self.poi_reps = nn.Embedding(num_pois, hist_dim)
        self.user_tower = UserTower(self.poi_reps, user_cat_dims, user_cont_dim, history_len, embed_dim, self.poi_cat_class)
        
        self.poi_tower = POITower(self.poi_reps, poi_cat_dims, poi_cont_dim, embed_dim, hist_dim)
        
        # self.user_embedding_dim = user_cont_dim + len(user_cat_dims) * self.embed_dim
        # self.item_embedding_dim = 256 + len(poi_cat_dims) * self.embed_dim + poi_cont_dim 

    
    def get_user_embedding(self, user_cat, user_cont, history, time, loc):
        user_vec, cat_logits =  self.user_tower(user_cat, user_cont, history, time, loc)
        return user_vec, cat_logits

    def get_poi_embedding(self, poi_idxs, poi_cat, poi_cont, poi_coords):
        return self.poi_tower(poi_idxs, poi_cat, poi_cont, poi_coords)
    
    
    def forward(self, user_cat, user_cont, trans_history, view_history, review_history,
                target_time, target_loc,
                poi_idxs, poi_cat, poi_cont, poi_coords, poi_mode='feat'):
        
        user_vec, cat_logits = self.user_tower(user_cat, user_cont, trans_history, view_history, review_history, target_time, target_loc)  # [B, D]
        
        if poi_mode == 'feat':
            poi_vec = self.poi_reps.weight  # [N, D]
        else:
            poi_vec = self.poi_tower(poi_idxs, poi_cat, poi_cont, poi_coords)            # [B, D]
                
        return user_vec, poi_vec, cat_logits


