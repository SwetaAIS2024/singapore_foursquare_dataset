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

    parser.add_argument("--epochs", "-e", type=int, default=30, help="Number of epochs (default: 30)")
    parser.add_argument("--print", type=int, default=50, help="Number of iterations to print loss")
    parser.add_argument("--batch_size", "-b", type=int, default=16, help="Batch size (default: 256)")
    parser.add_argument("--hidden_size", type=int, default=256, help="Transformer hidden size")
    parser.add_argument("--max_hist_length", type=int, default=200, help="Max history record length for each user")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Portion of training set")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Portion of validation set")
    parser.add_argument("--data_path", type=str, default="../data/SG_synthetic/app_profiles_all_users_version_4.json", help="Data path")
    parser.add_argument("--poi_path", type=str, default="../data/SG_synthetic/all_poi_id_all_interactions.json", help="POI data path")
    parser.add_argument("--model_path", type=str, default="../data/SG_synthetic/model_ckpt.pth", help="Model path")

    
    args = parser.parse_args()
    batch_size = args.batch_size

    
    user_data, poi_dict, user_dict = read_dataset(args.data_path, args.poi_path)
    
    user_cat_dims, user_cont_normalizers = process_user_info(user_dict)
    poi_cat_dims, poi_cont_normalizers = process_poi_info(poi_dict)
    num_pois = len(poi_dict) + 1 ## +1 for padding 
    print(user_cat_dims, user_cont_normalizers)
    print(num_pois, poi_cat_dims, poi_cont_normalizers)


    _, _, test_data = process_user_histories(user_data, args.train_ratio, args.val_ratio)
    
    test_dataset = TestDataset(test_data, user_dict, user_cont_normalizers, poi_dict, poi_cont_normalizers, args.max_hist_length)
    
    print("test len  =", len(test_dataset))

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    
    model = RecommendationModel(num_pois, user_cat_dims, poi_cat_dims, len(user_cont_normalizers), len(poi_cont_normalizers)-2) 
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(state_dict)
    model.to(args.device)
            
            
    print(f"Run model for test data")
    val_metric = evaluate_nxt(model, test_loader, args.device)

        

