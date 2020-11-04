"""
The inferance model
"""
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
import sklearn.metrics as metrics


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
        _, preds = torch.max(output, dim=1)
        return output, int(preds)
    

class SentimentRegressor(nn.Module):
    """
    BERT電影影評評分迴歸模型的主體
    Bert sentiment regression model for review sentiment analyzer
    """
    def __init__(self):
        super(SentimentRegressor, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.out = nn.Linear(self.bert.config.hidden_size, 1)

    def load_model(self, path):
        """
        載入先前訓練好的權重檔
        """
        self.load_state_dict(torch.load(path, map_location=DEVICE))

    def predicts(self, text):
        """
        主要的迴歸器，將input電影評論輸入模型，將輸出轉化為預測評分
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
        return float(self.out(output))


if __name__ == "__main__":
    VAL = pd.read_json("./data/test.json")
    TEST = list(VAL.comment)[:200] + list(VAL.comment)[-200:]
    TRUE = list(VAL.score)[:200] + list(VAL.score)[-200:]

    """ 分類模型 """
    MODEL = SentimentClassifier(11)
    MODEL.load_model("./regression_model/last_model_state.bin")
    prob, predict = MODEL.predicts(TEST[1].replace("<br />", " "))
    print(prob)
    data = []
    for test in tqdm(TEST):
        prob, predict = MODEL.predicts(test.replace("<br />", " "))
        print(prob)
        data.append(predict)
    with open('./data/test_result.txt', "r") as f:
        data = f.read().split("\n")
    data = [int(d) for d in data]

    metrics.accuracy_score(TRUE, data)
    metrics.r2_score(TRUE, data)

    RESULT = pd.DataFrame({"True": TRUE, "Predict": data})
    confusion_matrix = pd.crosstab(RESULT["True"], RESULT["Predict"], rownames=['Actual'], colnames=['Predicted'])
    sn.heatmap(confusion_matrix, annot=True)
    plt.show()

    """ 迴歸模型 """
    MODEL = SentimentRegressor()
    MODEL.load_model("./regression_model/last_model_state.bin")
    for test in tqdm(TEST):
        predict = MODEL.predicts(test.replace("<br />", " "))
        data.append(predict)
    with open('./data/test_result.txt', "r") as f:
        data = f.read().split("\n")
    data = [round(float(d)) for d in data]

    metrics.accuracy_score(TRUE, data)
    metrics.r2_score(TRUE, data)

    RESULT = pd.DataFrame({"True": TRUE, "Predict": data})
    confusion_matrix = pd.crosstab(RESULT["True"], RESULT["Predict"], rownames=['Actual'], colnames=['Predicted'])
    sn.heatmap(confusion_matrix, annot=True)
    plt.show()
