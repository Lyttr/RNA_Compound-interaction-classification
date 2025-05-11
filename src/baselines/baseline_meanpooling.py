
import pandas as pd
import torch
import pickle


train_df = pd.read_csv('datasets/trainset.csv')
test_df = pd.read_csv('datasets/testset.csv')


with open("raw/id_to_features_mean.pkl", "rb") as f:
    rna_dict = pickle.load(f)
with open("raw/id_to_features_compound.pkl", "rb") as f:
    compound_dict = pickle.load(f)


def create_dataset(df):
    dataset = []
    for _, row in df.iterrows():
        id1, id2 = row['Raw_ID1'], row['Raw_ID2']
        label = torch.tensor(row['label'])

        rna_feat = torch.tensor(rna_dict[id1]).unsqueeze(0) 
        smiles_feat = compound_dict[str(id2)][0].squeeze(0)     
        chem_feat = compound_dict[str(id2)][1]                   

        features = torch.cat([rna_feat, smiles_feat, chem_feat], dim=-1)
        dataset.append([features, label])
    return dataset

train_dataset = create_dataset(train_df)
test_dataset = create_dataset(test_df)


torch.save(train_dataset, 'datasets/trainset_meanpooling.pt')
torch.save(test_dataset, 'datasets/testset_meanpooling.pt')

print("Saved datasets to datasets/trainset_meanpooling.pt and datasets/testset_meanpooling.pt")