'''
python src/utils/fix_bug.py
''' 


'''
python3 main.py --min_clients_federation 2

python3 main.py --id 1 --source src.data/training_data/synthetic.npz
python3 main.py --id 2 --source src.ata/training_data/synthetic.npz

'''


# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# import torch
# from src.avitm.avitm import AVITM  # thay bằng tên class model của bạn
# from src.utils.utils import prepare_data_avitm_federated
# import numpy as np

# file = "src/data/training_data/synthetic.npz"
# data = np.load(file, allow_pickle=True)
# corpus = data['documents'][5-1]
# print("Số lượng phần tử trong data['documents']:", len(data['documents']))

# train_dataset, input_size, id2token = prepare_data_avitm_federated(corpus, 0.99, 0.01)

# model_parameters = {
#         "input_size": input_size,
#         "n_components": 10,
#         "model_type": "prodLDA",
#         "hidden_sizes": (100, 100),
#         "activation": "softplus",
#         "dropout": 0.2,
#         "learn_priors": True,
#         "batch_size": 64,
#         "lr": 2e-3,
#         "momentum": 0.99,
#         "solver": "adam",
#         "num_epochs": 100,
#         "reduce_on_plateau": False
#     }

# # 1. Khởi tạo model
# model = AVITM(input_size=model_parameters["input_size"],
#                   n_components=model_parameters["n_components"],
#                   model_type=model_parameters["model_type"],
#                   hidden_sizes=model_parameters["hidden_sizes"],
#                   activation=model_parameters["activation"],
#                   dropout=model_parameters["dropout"],
#                   learn_priors=model_parameters["learn_priors"],
#                   batch_size=model_parameters["batch_size"],
#                   lr=model_parameters["lr"],
#                   momentum=model_parameters["momentum"],
#                   solver=model_parameters["solver"],
#                   num_epochs=model_parameters["num_epochs"],
#                   reduce_on_plateau=model_parameters["reduce_on_plateau"])
    

# # 2. Load state_dict từ file .pth
# model.model.load_state_dict(torch.load("model.pth"))

# # 3. Đưa model về chế độ eval (nếu inference)
# model.eval()


import numpy as np
import torch
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Đọc file synthetic.npz
npz_path = "src/data/training_data/synthetic.npz"
data = np.load(npz_path, allow_pickle=True)
print("Các key trong synthetic.npz:", list(data.keys()))

# In thử 1 dòng đầu tiên của 'documents' và ghi ra file
# output_path = "src/utils/preview_documents_0.txt"
# with open(output_path, "w", encoding="utf-8") as f:
#     f.write("1 documents đầu tiên trong synthetic.npz:\n")
#     for i, doc in enumerate(data['documents'][:1]):
#         f.write(f"Document {i}: {doc}\n")
# print(f"Đã ghi nội dung vào file: {output_path}")

# In thử 1 dòng đầu tiên của 'n_nodes' và ghi ra file
output_path = "src/utils/n_nodes.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("1 n_nodes đầu tiên trong synthetic.npz:\n")
    f.write(f"n_nodes: {data['n_nodes']}\n")
print(f"Đã ghi nội dung vào file: {output_path}")

# In thử 1 dòng đầu tiên của 'n_nodes' và ghi ra file
output_path = "src/utils/vocab_size.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("1 vocab_size đầu tiên trong synthetic.npz:\n")
    f.write(f"vocab_size: {data['vocab_size']}\n")
print(f"Đã ghi nội dung vào file: {output_path}")

# In thử 1 dòng đầu tiên của 'n_nodes' và ghi ra file
output_path = "src/utils/n_topics.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("1 n_topics đầu tiên trong synthetic.npz:\n")
    f.write(f"n_topics: {data['n_topics']}\n")
print(f"Đã ghi nội dung vào file: {output_path}")

# In thử 1 dòng đầu tiên của 'n_nodes' và ghi ra file
output_path = "src/utils/frozen_topics.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("1 frozen_topics đầu tiên trong synthetic.npz:\n")
    f.write(f"frozen_topics: {data['frozen_topics']}\n")
print(f"Đã ghi nội dung vào file: {output_path}")

# In thử 1 dòng đầu tiên của 'n_nodes' và ghi ra file
output_path = "src/utils/preview_documents_5.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("1 beta đầu tiên trong synthetic.npz:\n")
    f.write(f"Beta: {data['beta']}\n")
print(f"Đã ghi nội dung vào file: {output_path}")

# In thử 1 dòng đầu tiên của 'n_nodes' và ghi ra file
output_path = "src/utils/beta.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("1 beta đầu tiên trong synthetic.npz:\n")
    f.write(f"beta: {data['beta']}\n")
print(f"Đã ghi nội dung vào file: {output_path}")

# In thử 1 dòng đầu tiên của 'n_nodes' và ghi ra file
output_path = "src/utils/alpha.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("1 alpha đầu tiên trong synthetic.npz:\n")
    f.write(f"alpha: {data['alpha']}\n")
print(f"Đã ghi nội dung vào file: {output_path}")

# In thử 1 dòng đầu tiên của 'n_nodes' và ghi ra file
output_path = "src/utils/preview_dn_docsocuments_8.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("1 n_docs đầu tiên trong synthetic.npz:\n")
    f.write(f"n_docs: {data['n_docs']}\n")
print(f"Đã ghi nội dung vào file: {output_path}")

# In thử 1 dòng đầu tiên của 'n_nodes' và ghi ra file
output_path = "src/utils/nwords.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("1 nwords đầu tiên trong synthetic.npz:\n")
    f.write(f"nwords: {data['beta']}\n")
print(f"Đã ghi nội dung vào file: {output_path}")

# In thử 1 dòng đầu tiên của 'n_nodes' và ghi ra file
output_path = "src/utils/topic_vectors.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("1 topic_vectors đầu tiên trong synthetic.npz:\n")
    f.write(f"topic_vectors: {data['topic_vectors']}\n")
print(f"Đã ghi nội dung vào file: {output_path}")

# In thử 1 dòng đầu tiên của 'n_nodes' và ghi ra file
output_path = "src/utils/doc_topics.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("1 doc_topics đầu tiên trong synthetic.npz:\n")
    f.write(f"doc_topics: {data['doc_topics']}\n")
print(f"Đã ghi nội dung vào file: {output_path}")

# In thử 1 dòng đầu tiên của 'n_nodes' và ghi ra file
output_path = "src/utils/z.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("1 z đầu tiên trong synthetic.npz:\n")
    f.write(f"z: {data['z']}\n")
print(f"Đã ghi nội dung vào file: {output_path}")



print("-" * 40)

# # Đọc file epoch_97.pth
# pth_path = "src/data/output_models/AVITM/epoch_97.pth"
# checkpoint = torch.load(pth_path, map_location='cpu')
# print("\nCác key trong epoch_97.pth:", list(checkpoint.keys()))

# # In thử 1 phần tử đầu tiên của state_dict nếu là tensor hoặc mảng
# if 'state_dict' in checkpoint:
#     print("\nCác tensor trong state_dict (chỉ in shape và vài giá trị đầu):")
#     for k, v in checkpoint['state_dict'].items():
#         print(f"{k}: shape={getattr(v, 'shape', None)}")
#         if hasattr(v, 'numpy'):
#             arr = v.cpu().numpy().flatten()
#             print("Giá trị đầu:", arr[:5])
#         elif hasattr(v, '__getitem__'):
#             print("Giá trị đầu:", v[:5])
#         print("-" * 20)

# # Nếu có key khác là dict hoặc list, in thử vài phần tử đầu
# for key in checkpoint.keys():
#     if key != 'state_dict':
#         val = checkpoint[key]
#         if isinstance(val, dict):
#             print(f"\nCác key trong '{key}':", list(val.keys())[:5])
#         elif isinstance(val, list):
#             print(f"\n5 phần tử đầu trong '{key}':", val[:5])
#         else:
#             print(f"\nGiá trị của '{key}':", str(val)[:200])





'''
pip install --upgrade protobuf==3.20.3 grpcio grpcio-tools

pip install 'fancycompleter<0.9'
pip install --upgrade pdbpp
'''
