import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
class TranslationDataset(Dataset):
    def __init__(self, path, tokenizer):
        self.df = pd.read_csv(path)
        self.tokenizer = tokenizer

        self.en = self.df['en'].tolist()
        self.bn = self.df['bn'].tolist()

    def __len__(self):
        return len(self.en)

    def __getitem__(self, idx):
        en = self.en[idx]
        bn = self.bn[idx]
        en_ids = self.tokenizer.encode(en, out_type=int)
        bn_ids = self.tokenizer.encode(bn, add_bos=True, add_eos=True, out_type=int)
        return torch.tensor(en_ids), torch.tensor(bn_ids)

