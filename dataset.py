import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import AutoTokenizer


class OffensiveDataset(Dataset):

    def __init__(self, sentence, label, tokenizer, max_len):
        self.sentence = sentence
        self.label = label
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, item):
        sentence = str(self.sentence[item])
        label = self.label[item]

        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'sentences': sentence,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def create_data_loader(dataframe, tokenizer, max_len, batch_size, shuffle):
    ds = OffensiveDataset(dataframe, tokenizer, max_len)
    return DataLoader(ds,
                      shuffle=False,
                      batch_size=batch_size,
                      num_workers=2)
