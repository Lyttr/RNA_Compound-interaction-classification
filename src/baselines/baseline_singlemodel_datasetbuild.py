import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

MAX_LEN = 1024
VOCAB_SIZE = 64
BATCH_SIZE = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def tokenizer(text: str, vocab_size: int) -> list:
    return [hash(c) % vocab_size for c in text]

class SMILESRNA_Dataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)

        self.texts = (df['SMILES'].astype(str) + df['RNA_seq'].astype(str)).tolist()
        self.labels = df['label'].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = tokenizer(self.texts[idx], VOCAB_SIZE)

        if len(tokens) > MAX_LEN:
            tokens = tokens[:MAX_LEN]
        else:
            tokens += [0] * (MAX_LEN - len(tokens))

        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def extract_tokenized_inputs(csv_path, save_path):
    dataset = SMILESRNA_Dataset(csv_path)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    output_dataset = []

    for batch in tqdm(loader, desc=f"Tokenizing {csv_path}"):
        input_ids = batch['input_ids'].to(DEVICE)
        label = batch['label'].to(DEVICE)

        output_dataset.append([
            input_ids.squeeze(0).cpu(),
            label.squeeze(0).cpu()
        ])

    torch.save(output_dataset, save_path)
    print(f"Saved tokenized data to {save_path} with {len(output_dataset)} samples.")


if __name__ == '__main__':
    os.makedirs('datasets', exist_ok=True)

    extract_tokenized_inputs('datasets/train_text.csv', 'datasets/trainset_baseline_singlemodel.pt')
    extract_tokenized_inputs('datasets/test_text.csv', 'datasets/testset_baseline_singlemodel.pt')