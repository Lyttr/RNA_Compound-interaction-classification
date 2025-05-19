import pandas as pd
import torch
import pickle
from torch_geometric.data import Data

# Load CSV and pickles
train_df = pd.read_csv('datasets/trainset.csv')

with open('raw/id_to_tokens.pkl', 'rb') as f:
    tokens_dict = pickle.load(f)
with open('raw/id_to_graph.pkl', 'rb') as f:
    graph_dict = pickle.load(f)
with open('raw/id_to_image.pkl', 'rb') as f:
    image_dict = pickle.load(f)
MAX_TOKEN_LENGTH = 1024  # for padding

def create_dataset(df):
    dataset = []
    for _, row in df.iterrows():
        id1, id2 = row['Raw_ID1'], row['Raw_ID2']
        label = torch.tensor(row['label'], dtype=torch.long)

        # --- Tokens ---
        tokens_feat = torch.tensor(tokens_dict[id1], dtype=torch.long)
        if tokens_feat.size(0) > MAX_TOKEN_LENGTH:
            tokens_feat = tokens_feat[:MAX_TOKEN_LENGTH]
        else:
            pad_len = MAX_TOKEN_LENGTH - tokens_feat.size(0)
            tokens_feat = torch.cat([tokens_feat, torch.ones(pad_len,dtype=torch.long)], dim=0)
        
        # --- Graph ---
        graph_data = graph_dict[str(id2)][0]
        image_data = image_dict[str(id2)][0]
        
        dataset.append((tokens_feat, graph_data,image_data, label))
    return dataset

# Build & save datasets
train_dataset = create_dataset(train_df)
print(len(train_df))
torch.save(train_dataset, 'datasets/trainset_baseline_vanllia.pt')

test_df = pd.read_csv('datasets/testset.csv')
test_dataset = create_dataset(test_df)
torch.save(test_dataset, 'datasets/testset_baseline_vanllia.pt')

print("Saved train/test datasets with token and graph pairs.")