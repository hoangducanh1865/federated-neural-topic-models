'''
python utils/fix_bug.py
''' 

# import numpy as np

# data = np.load("data/training_data/synthetic.npz", allow_pickle=True)
# print("Available keys:", list(data.keys()))


'''
python3 main.py --min_clients_federation 5

python3 main.py --id 1 --source data/training_data/synthetic.npz
python3 main.py --id 2 --source data/training_data/synthetic.npz
python3 main.py --id 3 --source data/training_data/synthetic.npz
python3 main.py --id 4 --source data/training_data/synthetic.npz
python3 main.py --id 5 --source data/training_data/synthetic.npz
'''


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from avitm.avitm import AVITM  # thay bằng tên class model của bạn
from utils.utils import prepare_data_avitm_federated
import numpy as np

file = "data/training_data/synthetic.npz"
data = np.load(file, allow_pickle=True)
corpus = data['documents'][5-1]
print("Số lượng phần tử trong data['documents']:", len(data['documents']))

train_dataset, input_size, id2token = prepare_data_avitm_federated(corpus, 0.99, 0.01)

model_parameters = {
        "input_size": input_size,
        "n_components": 10,
        "model_type": "prodLDA",
        "hidden_sizes": (100, 100),
        "activation": "softplus",
        "dropout": 0.2,
        "learn_priors": True,
        "batch_size": 64,
        "lr": 2e-3,
        "momentum": 0.99,
        "solver": "adam",
        "num_epochs": 100,
        "reduce_on_plateau": False
    }

# 1. Khởi tạo model
model = AVITM(input_size=model_parameters["input_size"],
                  n_components=model_parameters["n_components"],
                  model_type=model_parameters["model_type"],
                  hidden_sizes=model_parameters["hidden_sizes"],
                  activation=model_parameters["activation"],
                  dropout=model_parameters["dropout"],
                  learn_priors=model_parameters["learn_priors"],
                  batch_size=model_parameters["batch_size"],
                  lr=model_parameters["lr"],
                  momentum=model_parameters["momentum"],
                  solver=model_parameters["solver"],
                  num_epochs=model_parameters["num_epochs"],
                  reduce_on_plateau=model_parameters["reduce_on_plateau"])
    

# 2. Load state_dict từ file .pth
model.model.load_state_dict(torch.load("model.pth"))

# 3. Đưa model về chế độ eval (nếu inference)
model.eval()
