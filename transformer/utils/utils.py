import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    en_batch, bn_batch = zip(*batch)

    en_batch = pad_sequence(en_batch, batch_first=True, padding_value=0)
    bn_batch = pad_sequence(bn_batch, batch_first=True, padding_value=0)

    return en_batch, bn_batch
