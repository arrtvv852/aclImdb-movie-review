
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import BertModel, BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup


PRE_TRAINED_MODEL_NAME = "bert-base-cased"
BATCH_SIZE = 16
MAX_LEN = 160
EPOCHS = 10

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TOKENIZER = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)


def clean_corpus(corpus):
    """
    清除電影評論中的html元素 <br />
    do basic cleaning to the courpus, here only remove html content <br />
    """
    corpus.replace("<br />", " ")
    return corpus


class ReviewDataset(Dataset):
    """
    將資料集轉換為後續data DataLoader 需求的 pytorch Dataset形式
    Convert movie review dataframe into torch dataset instance
    """
    def __init__(self, comments, targets, max_len):
        self.comments = comments
        self.targets = targets
        self.max_len = max_len

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, item):
        comment = str(self.comments[item])
        target = self.targets[item]
        encoding = TOKENIZER.encode_plus(
            comment,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'comment': comment,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.float)
        }


def create_data_loader(dataframe, max_len, batch_size):
    """
    將pytorch Dataset形式資料集包裝為data DataLoader
    convert dataset to pytorch dataloader format object
    """
    dataset = ReviewDataset(
        comments=list(dataframe.comment.to_numpy()),
        targets=list(dataframe.score.to_numpy()),
        max_len=max_len
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        # num_workers=4
    )


class SentimentRegressor(nn.Module):
    """
    BERT電影影評評分回歸模型的主體
    Bert sentiment regression model for review sentiment analyzer
    """
    def __init__(self):
        super(SentimentRegressor, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.2)
        self.out = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)


class SentimentClassifier(nn.Module):
    """
    BERT電影影評評分分類模型的主體
    Bert sentiment classification model for review sentiment analyzer
    """
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.2)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)


def preds(model, text):
    """
    主樣的分類氣，將input電影評論輸入模型，將輸出轉化為預測評分
    make prediction according to the text with the given model
    """
    encoding = TOKENIZER.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    output = model.forward(encoding["input_ids"], encoding["attention_mask"])
    _, preds = torch.squeeze(output, dim=1)
    return int(preds)


def train_epoch(model,
                data_loader,
                loss_fn,
                optimizer,
                scheduler,
                n_examples):
    """
    電影評論分類器的訓練主流程
    Main training process of bert sentiment classifier
    """
    model = model.train()
    losses = []
    correct_predictions = 0
    for _d in data_loader:
        input_ids = _d["input_ids"].to(DEVICE)
        attention_mask = _d["attention_mask"].to(DEVICE)
        targets = _d["targets"].to(DEVICE)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        preds = torch.squeeze(outputs, dim=1)
        loss = loss_fn(preds, targets)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model,
               data_loader,
               loss_fn,
               n_examples):
    """
    電影評論分類器的訓練時每個epoch評估訓練效能主流程
    Main evaluate process in training of bert sentiment classifier
    """
    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(DEVICE)
            attention_mask = d["attention_mask"].to(DEVICE)
            targets = d["targets"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)


if __name__ == "__main__":
    TRAIN = pd.read_json("./data/train.json")
    TRAIN = TRAIN.sample(frac=1).reset_index(drop=True)
    TRAIN.comment = TRAIN.comment.apply(clean_corpus)
    VAL = pd.read_json("./data/test.json")
    VAL = VAL.sample(frac=1).reset_index(drop=True)
    VAL.comment = VAL.comment.apply(clean_corpus)
    TRAIN = TRAIN.append(VAL[500:]).reset_index(drop=True)
    VAL = VAL.iloc[:500]
    # sample_txt = df.comment[0]
    # train_data_loader = create_data_loader(df, MAX_LEN, BATCH_SIZE)
    TRAIN_DATA_LOADER = create_data_loader(TRAIN, MAX_LEN, BATCH_SIZE)
    VAL_DATA_LOADER = create_data_loader(VAL, MAX_LEN, BATCH_SIZE)
    # data = next(iter(train_data_loader))

    MODEL = SentimentRegressor()
    MODEL.to(DEVICE)

    # input_ids = data['input_ids'].to(DEVICE)
    # attention_mask = data['attention_mask'].to(DEVICE)
    # nn.functional.softmax(model(input_ids, attention_mask), dim=1)
    OPTIMIZER = AdamW(MODEL.parameters(), lr=2e-5, correct_bias=False)
    TOTAL_STEPS = len(TRAIN_DATA_LOADER) * EPOCHS
    SCHEDULER = get_linear_schedule_with_warmup(
        OPTIMIZER,
        num_warmup_steps=0,
        num_training_steps=TOTAL_STEPS
    )
    # LOSS_FN = nn.CrossEntropyLoss().to(DEVICE)
    LOSS_FN = nn.MSELoss().to(DEVICE)

    BEST_ACCURACY = 0

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(
            MODEL,
            TRAIN_DATA_LOADER,
            LOSS_FN,
            OPTIMIZER,
            SCHEDULER,
            len(TRAIN)
        )

        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, _ = eval_model(
            MODEL,
            VAL_DATA_LOADER,
            LOSS_FN,
            len(VAL)
        )

        print(f'Val accuracy {val_acc}')
        print()
        if val_acc > BEST_ACCURACY:
            torch.save(MODEL.state_dict(), 'best_model_state.bin')
            best_accuracy = val_acc
    torch.save(MODEL.state_dict(), 'last_model_state.bin')

