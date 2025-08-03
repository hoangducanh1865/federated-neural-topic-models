import torch
import numpy as np
from avitm.avitm import AVITM
from itertools import chain
import pickle
from eva.topic_coherence import compute_coherence
from eva.topic_diversity import compute_diversity

# ==== Load dữ liệu ====
data = np.load("data/training_data/synthetic.npz", allow_pickle=True)
corpus = data["documents"]

# Chuyển từng doc thành list từ đơn
flattened_corpus = [list(chain.from_iterable(doc)) for doc in corpus]

# ==== Load vocab từ file (đảm bảo thứ tự giống lúc train) ====
model_dir = "data/output_models/AVITM_nc_10_tpm_0.0_tpv_0.9_hs_prodLDA_ac_(100, 100)_do_softplus_lr_0.2_mo_0.002_rp_0.99"
with open(f"{model_dir}/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

# ==== Khởi tạo mô hình (cấu hình phải giống lúc train) ====
input_size = len(vocab)
n_components = 50  # phải đúng với lúc train

model = AVITM(
    input_size=input_size,
    n_components=n_components,
    model_type='prodLDA',
    hidden_sizes=(100, 100),
    activation='softplus',
    dropout=0.2,
    learn_priors=True,
    batch_size=64,
    lr=2e-3,
    momentum=0.99,
    solver='adam',
    num_epochs=100,
    reduce_on_plateau=False
)

# ==== Load checkpoint ====
epoch = 49
model.load(model_dir, epoch)

# ==== Gán lại vocab vào model để get_topics dùng được ====
model.train_data = type("FakeDataset", (), {})()
model.train_data.idx2token = vocab

# ==== Lấy top từ mỗi chủ đề ====
top_words = model.get_topics(k=10)

# ==== In top từ theo từng topic ====
print("\nTop words per topic:")
for topic_id, words in top_words.items():
    print(f"Topic {topic_id:2}: {', '.join(words)}")

# ==== Tính chỉ số đánh giá ====
tc = compute_coherence(top_words, reference_corpus=flattened_corpus, vocab=vocab)
td = compute_diversity(top_words)

print(f"\nTopic Coherence: {tc:.4f}")
print(f"Topic Diversity: {td:.4f}")