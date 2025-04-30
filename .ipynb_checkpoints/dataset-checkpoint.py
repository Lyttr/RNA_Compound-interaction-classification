# Step 1: Prepare CSVs (train/test split and text format)
import pandas as pd
import numpy as np
import os

os.makedirs('datasets', exist_ok=True)

# Load and clean compound data
compound = pd.read_csv('raw/compound.csv')
compound = compound[compound['SMILES'].notna() & (compound['SMILES'] != '')]
compound = compound[~compound['id'].duplicated(keep=False)]

# Load and clean RNA data
rna = pd.read_csv('raw/rna.csv')
rna = rna[(rna['seqence'] != 'no result') & (rna['seqence'].str.len() <= 1023)]
rna = rna[~rna['id'].duplicated(keep=False)]

# Load raw interaction data
raw = pd.read_csv('raw/Download_data_RC.txt', sep='\t', dtype=str)
raw = raw.dropna(subset=['Raw_ID1', 'Raw_ID2'])
raw['Raw_ID1'] = raw['Raw_ID1'].str.split(':').str[1]
raw['Raw_ID2'] = raw['Raw_ID2'].str.split(':').str[1]

# Filter only RNA-compound pairs existing in both cleaned datasets
rna_ids = set(rna['id'])
compound_ids = set(compound['id'])
filtered = raw[raw['Raw_ID1'].isin(rna_ids) & raw['Raw_ID2'].isin(compound_ids)]

# Split into strong and weak bindings
strong = filtered[filtered['strong'].notna()]
weak = filtered[filtered['strong'].isna()]

strong.to_csv('datasets/strong.csv', index=False)
weak.to_csv('datasets/weak.csv', index=False)

def generate_dataset(weak_df, strong_df, total_size=5000, random_seed=42):
    np.random.seed(random_seed)

    test_size = total_size // 5
    train_size = total_size - test_size

    train_pos_num = int(train_size * 0.2)
    train_neg_num = train_size - train_pos_num
    test_pos_num = int(test_size * 0.2)
    test_neg_num = test_size - test_pos_num

    all_pos = pd.concat([weak_df, strong_df], ignore_index=True).drop_duplicates()
    all_pos['label'] = 1

    test_pos = strong_df.drop_duplicates().sample(n=test_pos_num, random_state=random_seed)

    test_pos_pairs = set(tuple(x) for x in test_pos[['Raw_ID1', 'Raw_ID2']].values)
    remaining_strong = strong_df[~strong_df[['Raw_ID1', 'Raw_ID2']].apply(lambda x: tuple(x) in test_pos_pairs, axis=1)]
    train_pos_candidates = pd.concat([remaining_strong, weak_df], ignore_index=True).drop_duplicates()

    train_pos = train_pos_candidates.sample(n=train_pos_num, random_state=random_seed)

    used_pairs = set(tuple(x) for x in pd.concat([train_pos, test_pos])[['Raw_ID1', 'Raw_ID2']].values)

    def generate_negative_samples(existing_pairs, required_num, id_set1, id_set2):
        neg_samples = set()
        while len(neg_samples) < required_num:
            id1, id2 = np.random.choice(list(id_set1)), np.random.choice(list(id_set2))
            if id1 == id2:
                continue
            pair = (id1, id2)
            if pair in existing_pairs or pair in neg_samples:
                continue
            neg_samples.add(pair)
        return pd.DataFrame(list(neg_samples), columns=['Raw_ID1', 'Raw_ID2'])

    id1_set = set(all_pos['Raw_ID1'].unique())
    id2_set = set(all_pos['Raw_ID2'].unique())

    train_neg = generate_negative_samples(used_pairs, train_neg_num, id1_set, id2_set)
    used_pairs.update(tuple(x) for x in train_neg.values)
    test_neg = generate_negative_samples(used_pairs, test_neg_num, id1_set, id2_set)

    train_pos['label'] = 1
    train_neg['label'] = 0
    test_pos['label'] = 1
    test_neg['label'] = 0

    train_df = pd.concat([train_pos, train_neg], ignore_index=True).sample(frac=1, random_state=random_seed).reset_index(drop=True)
    test_df = pd.concat([test_pos, test_neg], ignore_index=True).sample(frac=1, random_state=random_seed).reset_index(drop=True)

    return train_df, test_df

# Calculate total size and generate datasets
total_size = (len(strong) + len(weak)) * 5 - 1
train_df, test_df = generate_dataset(weak, strong, total_size=total_size)

print(f"Train size: {len(train_df)} | Pos: {(train_df['label'] == 1).sum()} | Neg: {(train_df['label'] == 0).sum()}")
print(f"Test  size: {len(test_df)} | Pos: {(test_df['label'] == 1).sum()} | Neg: {(test_df['label'] == 0).sum()}")

# Save Raw_ID1, Raw_ID2, label
train_df[['Raw_ID1', 'Raw_ID2', 'label']].to_csv('datasets/trainset.csv', index=False)
test_df[['Raw_ID1', 'Raw_ID2', 'label']].to_csv('datasets/testset.csv', index=False)

# Create text datasets
def create_text_dataset(df, rna_df, compound_df, out_path):
    rna_dict = dict(zip(rna_df['id'], rna_df['seqence']))
    compound_dict = dict(zip(compound_df['id'], compound_df['SMILES']))
    
    records = []
    for _, row in df.iterrows():
        rid, cid, label = row['Raw_ID1'], row['Raw_ID2'], row['label']
        rna_seq = rna_dict.get(rid, None)
        smiles = compound_dict.get(cid, None)
        if rna_seq is None or smiles is None:
            continue
        text = f"{rna_seq} [SEP] {smiles}"
        records.append({
            'RNA_ID': rid,
            'Compound_ID': cid,
            'RNA_seq': rna_seq,
            'SMILES': smiles,
            'text': text,
            'label': label
        })
    
    pd.DataFrame(records).to_csv(out_path, index=False)

create_text_dataset(train_df, rna, compound, 'datasets/train_text.csv')
create_text_dataset(test_df, rna, compound, 'datasets/test_text.csv')