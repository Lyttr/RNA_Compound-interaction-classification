import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
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
            padding=False,
            truncation=False,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def extract_embeddings(csv_path, save_path):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    dataset = TextDataset(csv_path, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    output_dataset = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Extracting {csv_path}"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            label = batch['label']

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state.squeeze(0) 

         
            scalar_per_token = hidden_states.mean(dim=-1)  

         
            seq_len = scalar_per_token.shape[0]
            if seq_len >= MAX_LEN:
                scalar_per_token = scalar_per_token[:MAX_LEN]
            else:
                padding = torch.zeros(MAX_LEN - seq_len).to(DEVICE)
                scalar_per_token = torch.cat([scalar_per_token, padding], dim=0)

            output_dataset.append([scalar_per_token.cpu(), label.squeeze(0).cpu()])

    torch.save(output_dataset, save_path)
    print(f"Saved to {save_path} with {len(output_dataset)} samples.")


if __name__ == '__main__':
    os.makedirs('datasets', exist_ok=True)

    extract_embeddings('datasets/train_text.csv', 'datasets/trainset_transformer.pt')
    extract_embeddings('datasets/test_text.csv', 'datasets/testset_transformer.pt')