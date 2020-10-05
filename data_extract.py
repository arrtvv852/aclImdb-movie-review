
import os
import json


data_path = "./data/aclImdb"

train_pos_files = os.listdir(os.path.join(data_path, "train/pos"))
train_neg_files = os.listdir(os.path.join(data_path, "train/neg"))

datas = []
for pos_file in train_pos_files:
    with open(os.path.join(data_path, "train/pos", pos_file), "r") as f:
        comment = f.read()
    score = int(pos_file.split("_")[1].split(".")[0])
    datas.append({"comment": comment, "score": score})

for neg_file in train_neg_files:
    with open(os.path.join(data_path, "train/neg", neg_file), "r") as f:
        comment = f.read()
    score = int(neg_file.split("_")[1].split(".")[0])
    datas.append({"comment": comment, "score": score})

with open("./data/train.json", "w") as f:
    f.write(json.dumps(datas))

test_pos_files = os.listdir(os.path.join(data_path, "test/pos"))
test_neg_files = os.listdir(os.path.join(data_path, "test/neg"))

datas = []
for pos_file in test_pos_files:
    with open(os.path.join(data_path, "test/pos", pos_file), "r") as f:
        comment = f.read()
    score = int(pos_file.split("_")[1].split(".")[0])
    datas.append({"comment": comment, "score": score})

for neg_file in test_neg_files:
    with open(os.path.join(data_path, "test/neg", neg_file), "r") as f:
        comment = f.read()
    score = int(neg_file.split("_")[1].split(".")[0])
    datas.append({"comment": comment, "score": score})

with open("./data/test.json", "w") as f:
    f.write(json.dumps(datas))
