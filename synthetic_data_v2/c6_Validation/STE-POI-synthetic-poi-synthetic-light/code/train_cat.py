import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR


from model import UserTower, POITower, RecommendationModel
from utils import *
from dataset import TrainDataset, TestDataset, collate_fn

import argparse



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Training Script")

    parser.add_argument("--epochs", "-e", type=int, default=50, help="Number of epochs (default: 30)")
    parser.add_argument("--print", type=int, default=50, help="Number of iterations to print loss")
    parser.add_argument("--batch_size", "-b", type=int, default=16, help="Batch size (default: 256)")
    parser.add_argument("--hidden_size", type=int, default=64, help="Transformer hidden size")
    parser.add_argument("--max_hist_length", type=int, default=200, help="Max history record length for each user")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Portion of training set")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Portion of validation set")
    parser.add_argument("--data_path", type=str, default="../data/SG_synthetic/app_profiles_all_users_version_5.json", help="Data path")
    parser.add_argument("--poi_path", type=str, default="../data/SG_synthetic/all_poi_id_all_interactions.json", help="POI data path")
    

    args = parser.parse_args()
    batch_size = args.batch_size

    
    user_data, poi_dict, user_dict, cat2idx = read_dataset(args.data_path, args.poi_path)
    
    user_cat_dims, user_cont_normalizers = process_user_info(user_dict)
    poi_cat_dims, poi_cont_normalizers = process_poi_info(poi_dict)
    num_pois = len(poi_dict) + 1 ## +1 for padding 
    print(user_cat_dims, user_cont_normalizers)
    print(num_pois, poi_cat_dims, poi_cont_normalizers)
    idx2cat = {idx:cat for cat,idx in cat2idx.items()}

    train_data, val_data, test_data, cat_freq = process_user_histories(user_data, args.train_ratio, args.val_ratio)
    
    train_dataset = TrainDataset(train_data, user_dict, user_cont_normalizers, poi_dict, poi_cont_normalizers, args.max_hist_length)
    val_dataset = TestDataset(val_data, user_dict, user_cont_normalizers, poi_dict, poi_cont_normalizers, args.max_hist_length)
    test_dataset = TestDataset(test_data, user_dict, user_cont_normalizers, poi_dict, poi_cont_normalizers, args.max_hist_length)
    # val_dataset = TestDataset([[records[0], records[1][:-1]] for records in user_data], user_dict, user_cont_normalizers, poi_dict, poi_cont_normalizers, args.max_hist_length)
    # test_dataset = TestDataset([[records[0], records[1]] for records in user_data], user_dict, user_cont_normalizers, poi_dict, poi_cont_normalizers, args.max_hist_length)
    
    print("train len =", len(train_dataset))
    print("val len   =", len(val_dataset))
    print("test len  =", len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    print(f"num of validation {len(val_dataset)}, num of test {len(test_dataset)}")

    freq_vec = torch.ones(len(cat2idx), dtype=torch.float32)
    for c, cnt in cat_freq.items():
        freq_vec[c] = cnt + 1
    class_weights = 1.0 / freq_vec
    class_weights = class_weights / class_weights.sum() * len(cat2idx)

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RecommendationModel(num_pois, user_cat_dims, poi_cat_dims, len(user_cont_normalizers), len(poi_cont_normalizers)-2, hist_dim=64) 
    # check_model(model)
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=3e-4)
    steps_per_epoch = len(train_loader)
    lr_lambda = cosine_with_warmup_schedule(steps_per_epoch, args.epochs, warmup_epochs=3)
    # criterion_cat_cls = nn.CrossEntropyLoss(weight=class_weights.to(args.device))
    # criterion_cat_cls = nn.CrossEntropyLoss(label_smoothing=0.1)
    # criterion_cat_cls = nn.CrossEntropyLoss()
    criterion_cat_cls = FocalLoss(
            gamma=2.0,
            reduction='mean'
        )
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=1e-6)

    model.to(args.device)

    losses = []
    best_val_score = 0

    for epoch in range(args.epochs):
        model.train()
        print(f"{epoch}/{args.epochs} Running!!, num of training samples {len(train_dataset)}")
        running_loss = 0.0
        for batch_no, batch in enumerate(train_loader):
            # print(batch)
            batch = move_to_device(batch, args.device)
            user_cat, user_cont = batch['user_cat'], batch['user_cont']
            poi_cat, poi_cont, poi_coords = batch['poi_cat'], batch['poi_cont'], batch['poi_coords']
            target_poi_idxs, target_time, target_loc, target_cat = batch['target_poi'], batch['target_time'], batch['target_coords'], batch['target_cat']
            
            # history = batch['history']
            trans_history  = batch['history']    
            view_history   = batch['view_history']
            review_history = batch['review_history']
            # print(f"trans_history {trans_history['loc_cats']}")
            # print(f"view_history {view_history['loc_cats']}")
            # print(f"review_history {review_history['loc_cats']}")
            
            optimizer.zero_grad()
            user_emb, item_emb, cat_logits = model(user_cat, user_cont, trans_history, view_history, review_history, target_time, target_loc,
                                        target_poi_idxs, poi_cat, poi_cont, poi_coords)
            
            # loss_poi = compute_loss(user_emb, item_emb)
            # loss_poi = F.cross_entropy(user_emb @ item_emb.t(), target_poi_idxs.squeeze())
            loss_cat = criterion_cat_cls(cat_logits, target_cat)
            
            # loss = loss_poi + loss_cat
            loss = loss_cat
        
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_ = loss.item()
            losses.append(loss_)
            running_loss += loss_
            
            if batch_no % args.print == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}, Iter {batch_no}, Loss {loss_:.4f}, Cat Loss {loss_cat.item():.4f}, Avg Loss {running_loss/(batch_no+1):.4f}, LR: {lr:.6f}")
                
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")
        
        # full_poi_embeds = get_all_poi_embedding(model, poi_dict, poi_cont_normalizers, args.device)
        # print(f"poi embedding shape: {full_poi_embeds.shape}")
        
        # val_metric = evaluate_nxt(model, val_loader, args.device)
        # val_metric = evaluate_nxt(model, val_loader, args.device, idx2cat)
        val_metric = evaluate_nxt_cat(model, val_loader, args.device, idx2cat)
       
        if val_metric > best_val_score:
            print("\n=== Best Validation Result ===")
            torch.save(model.state_dict(), f'../saved_model/recommendation_model_{epoch}_cat.pth')
            best_val_score = val_metric
            
            
            print(f"Run model for test data")
            # val_metric = evaluate_nxt(model, test_loader, args.device)
            # val_metric = evaluate_nxt(model, test_loader,args.device, idx2cat)
            val_metric = evaluate_nxt_cat(model, test_loader,args.device, idx2cat)

            
    print(f"Model Trained Successfully. Exiting with Code 0")


