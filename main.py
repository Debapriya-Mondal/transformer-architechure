import torch
from transformer.transformer import Transformer
from transformer.utils.dataset import TranslationDataset
from transformer.utils.utils import collate_fn
from transformer.train import TransformerTrain
from transformer.prediction import TransformerPredictor
import sentencepiece as spm
from torch.utils.data import DataLoader

def main():
    tokenizer = spm.SentencePieceProcessor(model_file='tokenizer/en_bn_spm.model')
    
    model = Transformer(
        tokenizer=tokenizer,
        d_model=512,
        num_heads=4, 
        num_layers=6
    )
    
    train = TranslationDataset("dataset/train_en_bn.csv", tokenizer)
    train = torch.utils.data.Subset(train, indices=range(0, 5000))
    print("size of data set: ", len(train))

    train_loader = DataLoader(train, batch_size=32, shuffle=True, collate_fn=collate_fn)
    print("train_loader size: ", len(train_loader))
    x, y = next(iter(train_loader))
    print("x shape: ", x.shape)
    print("y shape: ", y.shape)

    trainer = TransformerTrain(tokenizer, model, lr=3e-4)
    trainer.train(train_loader, 3)
    trainer.save("weights.pt")

    
    # loaded_state_dict = torch.load('weights.pt', weights_only=True)
    # model.load_state_dict(loaded_state_dict)

    predictor = TransformerPredictor(tokenizer, model)
    result = predictor.prediction("Dog is animal")
    print("prediction result: ", result)

if __name__ == "__main__":
    main()
