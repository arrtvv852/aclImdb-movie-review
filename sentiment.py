
import torch
import transformers
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup


PRE_TRAINED_MODEL_NAME = "bert-base-cased"
BATCH_SIZE = 16
MAX_LEN = 160

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TOKENIZER = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
BERT_MODEL = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

def clean_corpus(corpus):
    """ do basic cleaning to the courpus, here only remove html content <br /> """
    corpus.replace("<br />", " ")
    return corpus


class ReviewDataset(Dataset):
    """
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
            'targets': torch.tensor(target, dtype=torch.long)
        }


def create_data_loader(df, max_len, batch_size):
    """ convert dataset to pytorch dataloader format object """
    dataset = ReviewDataset(
        comments=list(df.comment.to_numpy()),
        targets=list(df.score.to_numpy()),
        max_len=max_len
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4
    )


class SentimentClassifier(nn.Module):
    """ Bert sentiment main model for review sentiment analyzer """
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

def train_epoch(model,
                data_loader,
                loss_fn,
                optimizer,
                device,
                scheduler,
                n_examples):
    """ Main training process of bert sentiment classifier """
    model = model.train()
    losses = []
    correct_predictions = 0
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return correct_predictions.double() / n_examples, np.mean(losses)


if __name__ == "__main__":
    df = pd.read_json("./data/train.json")
    df.comment = df.comment.apply(clean_corpus)
    # sample_txt = df.comment[0]
    # train_data_loader = create_data_loader(df, MAX_LEN, BATCH_SIZE)
    train_data_loader = create_data_loader(df, MAX_LEN, BATCH_SIZE)
    data = next(iter(train_data_loader))

    model = SentimentClassifier(len(range(1, 11)))
    model.to(DEVICE)

    # input_ids = data['input_ids'].to(DEVICE)
    # attention_mask = data['attention_mask'].to(DEVICE)
    # nn.functional.softmax(model(input_ids, attention_mask), dim=1)
