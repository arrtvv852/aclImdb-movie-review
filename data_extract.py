
import os
import json


data_path = "./data/aclImdb"

"""
存取訓練資料的檔案名稱,格式為 "<評論ID>_<評分>.txt" 如. 3223_8.txt
"""
train_pos_files = os.listdir(os.path.join(data_path, "train/pos"))
train_neg_files = os.listdir(os.path.join(data_path, "train/neg"))

"""
逐一讀取訓練資料正(pos)負(neg)評價中各個檔案,並且將評論以及評分除存為以下格式:
    [{"comment": "This was one of ...", "score": 8},
     {"comment": ..., "score": 7},
     {"comment": ..., "score": 10},
     ...
    ]
"""
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

""" 最後以json檔的格式儲存該訓練資料 """
with open("./data/train.json", "w") as f:
    f.write(json.dumps(datas))


"""
存取測試資料的檔案名稱,格式為 "<評論ID>_<評分>.txt" 如. 3223_8.txt
"""
test_pos_files = os.listdir(os.path.join(data_path, "test/pos"))
test_neg_files = os.listdir(os.path.join(data_path, "test/neg"))

"""
逐一讀取測試資料正(pos)負(neg)評價中各個檔案,並且將評論以及評分除存為以下格式:
    [{"comment": "This was one of ...", "score": 8},
     {"comment": ..., "score": 7},
     {"comment": ..., "score": 10},
     ...
    ]
"""
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

""" 最後以json檔的格式儲存該訓練資料 """
with open("./data/test.json", "w") as f:
    f.write(json.dumps(datas))
