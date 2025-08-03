import numpy as np

data = np.load("data/training_data/synthetic.npz", allow_pickle=True)
print("Available keys:", list(data.keys()))