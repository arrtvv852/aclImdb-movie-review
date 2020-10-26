"""
The inferance model
"""
import pandas as pd
import torch
from torch import nn
from transformers import BertModel, BertTokenizer


PRE_TRAINED_MODEL_NAME = "bert-base-cased"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TOKENIZER = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
MAX_LEN = 160


class SentimentClassifier(nn.Module):
    """
    BERT電影影評評分分類模型的主體
    Bert sentiment main model for review sentiment analyzer
    """
    def __init__(self, n_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.2)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def load_model(self, path):
        """
        載入先前訓練好的權重檔
        """
        self.load_state_dict(torch.load(path, map_location=DEVICE))


    def predicts(self, text):
        """
        主要的分類器，將input電影評論輸入模型，將輸出轉化為預測評分
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
        _, output = self.bert(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"]
        )
        output = nn.functional.softmax(
            self.out(output), dim=1)
        # output = self.out(output)
        _, preds = torch.max(output, dim=1)
        return output, int(preds)

if __name__ == "__main__":
    MODEL = SentimentClassifier(11)
    MODEL.load_model("./models/pytorch_model.bin")
    VAL = pd.read_json("./data/test.json")
    TEST = VAL.comment[:20] 
    PRED = []
    for test in TEST:
        prob, predict = MODEL.predicts(test.replace("<br />", " "))
        print(prob)
        PRED.append(predict)
