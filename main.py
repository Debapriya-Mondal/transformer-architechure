from transformer.utils.dataset import TranslationDataset
from transformer.utils.utils import collate_fn
import sentencepiece as spm
from torch.utils.data import DataLoader
def main():
    tokenizer = spm.SentencePieceProcessor(model_file='tokenizer/en_bn_spm.model')
    train = TranslationDataset("dataset/train_en_bn.csv", tokenizer)
    print("size of data set: ", len(train))

    train_loader = DataLoader(train, batch_size=2, shuffle=True, collate_fn=collate_fn)
    print("train_loader size: ", len(train_loader))
    x, y = next(iter(train_loader))
    print("x: ", x)
    print("y: ", y)


if __name__ == "__main__":
    main()
