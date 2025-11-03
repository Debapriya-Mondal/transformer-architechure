import torch

class TransformerPredictor:
    def __init__(self, tokenizer, model, max_len=10):
        self.tokenizer = tokenizer
        self.model = model.eval()
        self.max_len = max_len
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    def predict(self, x):
        with torch.no_grad():
            encoder_output = self.model.encoder(x, src_padding_mask=None)  # [1, src_len, d_model]
            bos = torch.tensor([[self.tokenizer.bos_id()]], device=x.device)
            eos = self.tokenizer.eos_id()
            generated = bos
            for _ in range(self.max_len):
                logits = self.model.decoder(generated, encoder_output, trgt_padding_mask=None, enc_padding_mask=None)
                next_token = logits[:, -1, :].argmax(-1, keepdim=True)
                if next_token.item() == eos:
                    break
                generated = torch.cat([generated, next_token], dim=1)
            return generated

    def decode(self, pred):
        text = ""
        for i in pred[0]:
            text = text + self.tokenizer.decode(i.item()) + " "
        return text

    def prediction(self, text):
        tokens = self.tokenizer.encode(text)
        tokens = torch.tensor(tokens).unsqueeze(0).to(self.device)
        pred = self.predict(tokens)
        return self.decode(pred)
