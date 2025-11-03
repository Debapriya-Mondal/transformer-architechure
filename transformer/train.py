import torch
import torch.nn as nn
from tqdm.auto import tqdm

class TransformerTrain:
    def __init__(self, tokenizer, model, lr=0.001):
        self.tokenizer = tokenizer
        self.model = model.train()
        self.lr = lr
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def train_single_epoch(self, data_loader, optimizer, criterion):

        total_loss = 0.0

        batches = tqdm(data_loader, desc="Batches", leave=False)

        for batch in batches:
            # batch["src"] : [B, src_len]
            # batch["tgt"] : [B, tgt_len]
            src = batch[0].to(self.device)
            tgt = batch[1].to(self.device)

            # Teacher forcing
            # decoder input = BOS ... last-1
            # target labels  = 2nd ... EOS
            decoder_in  = tgt[:, :-1]           # feeding BOS, tokens, but not final EOS
            gold_labels = tgt[:, 1:].contiguous().view(-1)  # shift left for prediction

            logits = self.model(src, decoder_in)     # [B, dec_len-1, vocab]
            logits = logits.reshape(-1, logits.size(-1))  # flatten for CrossEntropy

            loss = criterion(logits, gold_labels)

            optimizer.zero_grad()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            # scheduler.step()  (optional)

            total_loss += loss.item()

            batches.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(data_loader)
        return avg_loss


    def train(self, train_loader, epochs=10):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_id())
        for epoch in range(epochs):
            avg_loss = self.train_single_epoch(train_loader, optimizer, criterion)
            print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f}")

    def save(self, path):
        torch.save(self.model.state_dict(), path)


