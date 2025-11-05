import os
import json
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

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
        self.folder_name = None
        self.val_loader = None

    def warmup_scheduler(self, optimizer, d_model, warmup_steps):
        def lr_lambda(step):
            step = max(step, 1)
            return (d_model ** -0.5) * min(
                step ** -0.5,
                step * (warmup_steps ** -1.5)
            )
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

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=[0.9, 0.98], eps=1e-9)
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_id(), label_smoothing=0.1)

        scheduler = self.warmup_scheduler(optimizer, self.d_model, warmup_steps)

        self.val_loader=val_loader


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
            
            current_lr = optimizer.param_groups[0]['lr']

            self.history['epochs'].append(epoch+1)


            if val_loader is not None:
                val_loss = self.validate_single_epoch(val_loader, criterion)
                self.avg_val_loss = val_loss
                print(f"Epoch {epoch+1}/{epochs} - LR: {current_lr:.8f} - train: {train_loss:.4f} - val: {val_loss:.4f}")
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                
            else:
                print(f"Epoch {epoch+1}/{epochs} - LR: {current_lr:.8f} - train: {train_loss:.4f}")
                self.history['train_loss'].append(train_loss)

        self.save()
        self.plot_loss()


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

        self.folder_name = os.path.join("weights", f"run_id_{trained_at}")
        file_name = os.path.join(self.folder_name, f"model_weights.pt")
        os.makedirs(self.folder_name, exist_ok=True)

        torch.save(self.model.state_dict(), file_name)
        with open(os.path.join(self.folder_name, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

        df = pd.DataFrame(self.history)
        df.to_csv(os.path.join(self.folder_name, "training_log.csv"), index=False)

        print(f"model save in {self.folder_name}")

    def plot_loss(self):
        plt.figure(figsize=(8,5))
        if self.val_loader:
            plt.plot(self.history["epochs"], self.history["train_loss"], label="Train Loss", color="red")
            plt.plot(self.history["epochs"], self.history["val_loss"], label="Val Loss", color="blue")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.title("Training vs Validation Loss")
            plt.grid(True)
            plt.savefig(os.path.join(self.folder_name, "loss_plot.png"))
            plt.show()
        else:
            plt.plot(self.history["epochs"], self.history["train_loss"], label="Train Loss", color="red")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.title("Training Loss")
            plt.grid(True)
            plt.savefig(os.path.join(self.folder_name, "loss_plot.png"))
            plt.show()

        


