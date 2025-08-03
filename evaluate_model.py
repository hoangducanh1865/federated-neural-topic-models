import torch
import numpy as np
import os
from avitm import avitm
from eva.topic_coherence import topic_coherence
from eva.topic_diversity import topic_diversity

# Đường dẫn tới model và dữ liệu
model_path = "data/output_models/AVITM_nc_10_tpm_0.0_tpv_0.9_hs_prodLDA_ac_(100, 100)_do_softplus_lr_0.2_mo_0.002_rp_0.99/epoch_None.pth"
data_path = "data/training_data/train_data.npz"
save_result_path = "evaluation_results.txt"

# Load dữ liệu đã train
data = np.load(data_path, allow_pickle=True)
train_corpus = data["corpus"]
vocab = data["vocab"]  # vocab[i] là từ thứ i

# Load mô hình đã huấn luyện
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
model = AVITM(**checkpoint["config"])  # khởi tạo đúng theo config đã lưu
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Lấy ma trận chủ đề-từ (beta)
beta = model.get_beta().detach().cpu().numpy()  # shape: [n_topics, vocab_size]

# Đánh giá
coherence = topic_coherence(beta, train_corpus, vocab, topk=10)
diversity = topic_diversity(beta, topk=25)

# Hiển thị top từ trong mỗi topic
top_words = []
for topic_weights in beta:
    top_idx = topic_weights.argsort()[::-1][:10]
    words = [vocab[i] for i in top_idx]
    top_words.append(words)

# Ghi kết quả ra file
with open(save_result_path, "w", encoding="utf-8") as f:
    f.write("Top words of each topic:\n")
    for i, topic in enumerate(top_words):
        f.write(f"Topic {i}: {', '.join(topic)}\n")

    f.write(f"\nTopic Coherence: {coherence:.4f}\n")
    f.write(f"Topic Diversity: {diversity:.4f}\n")

print("Saved results in:", save_result_path)