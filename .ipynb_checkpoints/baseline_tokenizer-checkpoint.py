import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import os

MODEL_NAME = 'google/bigbird-roberta-base'
MAX_LEN = 1452
BATCH_SIZE = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class TextDataset(Dataset):
    def __init__(self, csv_file, tokenizer):
        df = pd.read_csv(csv_file)
        self.texts = df['text'].tolist()
        self.labels = df['label'].tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=MAX_LEN,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),   
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def extract_tokenized_inputs(csv_path, save_path):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    vocab_size = tokenizer.vocab_size 
    print(f"Tokenizer vocab size = {vocab_size}")

    dataset = TextDataset(csv_path, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    output_dataset = []

    for batch in tqdm(loader, desc=f"Tokenizing {csv_path}"):
        input_ids = batch['input_ids'].to(DEVICE).float()  
        label = batch['label']

        input_ids_norm = input_ids / (vocab_size - 1)  

        output_dataset.append([
            input_ids_norm.squeeze(0).cpu(), 
            label.squeeze(0).cpu()             
        ])

    torch.save(output_dataset, save_path)
    print(f"Saved normalized tokenized data to {save_path} with {len(output_dataset)} samples.")


if __name__ == '__main__':
    os.makedirs('datasets', exist_ok=True)

    extract_tokenized_inputs('datasets/train_text.csv', 'datasets/trainset_tokenized.pt')
    extract_tokenized_inputs('datasets/test_text.csv', 'datasets/testset_tokenized.pt')