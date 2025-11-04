import os
import json
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from datetime import datetime

class TransformerTrain:
    def __init__(self, tokenizer, model, lr=0.001):
        self.tokenizer = tokenizer
        self.model = model.train()
        self.lr = lr
        self.epochs = None
        self.avg_train_loss = None
        self.avg_val_loss = None
        self.trained_at = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.d_model = model.d_model
        self.num_heads = model.num_heads
        self.num_layers = model.num_layers

        self.history = {}

    def warmup_scheduler(self, optimizer, warmup_steps, total_steps):
        """
        Creates a LambdaLR scheduler with warmup + linear decay.
        """
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def train_single_epoch(self, data_loader, optimizer, scheduler, criterion):

        total_loss = 0.0

        batches = tqdm(data_loader, desc="Train Batches", leave=False)

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
            scheduler.step()

            total_loss += loss.item()

            batches.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(data_loader)
        return avg_loss
    
    def validate_single_epoch(self, val_loader, criterion):
        self.model.eval()
        total_loss = 0.0
        batches = tqdm(val_loader, desc="Val Batches", leave=False)

        with torch.no_grad():
            for batch in batches:
                src = batch[0].to(self.device)
                tgt = batch[1].to(self.device)

                decoder_in = tgt[:, :-1]
                gold_labels = tgt[:, 1:].contiguous().view(-1)

                logits = self.model(src, decoder_in)
                logits = logits.reshape(-1, logits.size(-1))

                loss = criterion(logits, gold_labels)
                total_loss += loss.item()
                batches.set_postfix(val_loss=loss.item())

        return total_loss / len(val_loader)



    def train(self, train_loader, val_loader=None, epochs=10, warmup_steps=4000):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_id())

        total_steps = epochs * len(train_loader)
        scheduler = self.warmup_scheduler(optimizer, warmup_steps, total_steps)

        self.epochs = epochs

        if val_loader:
            self.history = {
                "epochs": [],
                "train_loss": [],
                "val_loss": []
            }
        else:
            self.history = {
                "epochs": [],
                "train_loss": []
            }


        for epoch in range(epochs):
            train_loss = self.train_single_epoch(train_loader, optimizer, scheduler, criterion)
            self.avg_train_loss = train_loss

            self.history['epochs'].append(epoch+1)

            if val_loader is not None:
                val_loss = self.validate_single_epoch(val_loader, criterion)
                self.avg_val_loss = val_loss
                print(f"Epoch {epoch+1}/{epochs} - train: {train_loss:.4f} - val: {val_loss:.4f}")
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                
            else:
                print(f"Epoch {epoch+1}/{epochs} - train: {train_loss:.4f}")
                self.history['train_loss'].append(train_loss)

        self.save()


    def save(self):
        trained_at= datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        metadata = {
            "d_model":self.d_model,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "epochs": self.epochs,
            "avg_train_loss": self.avg_train_loss,
            "avg_val_loss": self.avg_val_loss,
            "device": self.device,
            "trained_at": trained_at
        }

        folder_name = os.path.join("weights", f"run_id_{trained_at}")
        file_name = os.path.join(folder_name, f"model_weights.pt")
        os.makedirs(folder_name, exist_ok=True)

        torch.save(self.model.state_dict(), file_name)
        with open(os.path.join(folder_name, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

        df = pd.DataFrame(self.history)
        df.to_csv(os.path.join(folder_name, "training_log.csv"), index=False)

        print(f"model save in {folder_name}")

        


