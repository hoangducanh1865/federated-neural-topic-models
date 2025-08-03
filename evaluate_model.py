import os
import torch
import numpy as np

from eva.topic_coherence import compute_coherence
from eva.topic_diversity import compute_diversity
from utils.utils import split_text_word

# ==== Đường dẫn tới dữ liệu và model ====
model_path = "data/output_models/AVITM_nc_10_tpm_0.0_tpv_0.9_hs_prodLDA_ac_(100, 100)_do_softplus_lr_0.2_mo_0.002_rp_0.99/epoch_None.pth"
data_path = "data/training_data/synthetic.npz"
save_result_path = "evaluation_results.txt"

# ==== Load dữ liệu ====
data = np.load(data_path, allow_pickle=True)
corpus = data["corpus"].tolist()
vocab = data["vocab"].tolist()

# ==== Load mô hình đã huấn luyện ====
checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
model = checkpoint["model"]
model.eval()

# ==== Lấy topic-word distributions ====
# Chú ý: model cần có phương thức get_beta(), hoặc bạn sửa lại nếu nó khác
try:
    beta = model.get_beta().detach().cpu().numpy()  # shape: [n_topics, vocab_size]
except AttributeError:
    raise ValueError("Model không có phương thức get_beta(). Hãy chắc rằng bạn đang dùng AVITM hoặc ProdLDA đúng cách.")

# ==== Lấy top từ của từng chủ đề ====
top_words = []
for topic_weights in beta:
    top_idx = topic_weights.argsort()[::-1][:10]
    words = [vocab[i] for i in top_idx]
    top_words.append(words)

# ==== Tính coherence và diversity ====
coherence = compute_coherence(reference_corpus=corpus, vocab=vocab, top_words=top_words)
diversity = compute_diversity(top_words)

# ==== Ghi kết quả ra file ====
with open(save_result_path, "w", encoding="utf-8") as f:
    f.write("Top từ mỗi topic:\n")
    for i, topic in enumerate(top_words):
        f.write(f"Topic {i}: {', '.join(topic)}\n")
    f.write("\n")
    f.write(f"Topic Coherence (c_v): {coherence:.4f}\n")
    f.write(f"Topic Diversity: {diversity:.4f}\n")

print(f"✅ Đánh giá hoàn tất. Kết quả được lưu vào: {save_result_path}")