import numpy as np


GT_ROOT = 'D:/UIT/Research/VPR-OT/salad/datasets/'

# Load thử từng file
dbImages = np.load(GT_ROOT + 'msls_val/msls_val_dbImages.npy')
qIdx = np.load(GT_ROOT + 'msls_val/msls_val_qIdx.npy')
qImages = np.load(GT_ROOT + 'msls_val/msls_val_qImages.npy')
ground_truth = np.load(GT_ROOT + 'msls_val/msls_val_pIdx.npy', allow_pickle=True)

# In thông tin cơ bản
print("dbImages shape:", dbImages.shape)
print("dbImages example:", dbImages[:5])   # in thử 5 phần tử đầu

print("qIdx shape:", qIdx.shape)
print("qIdx example:", qIdx[:10])

print("qImages shape:", qImages.shape)
print("qImages example:", qImages[:5])

print("ground_truth type:", type(ground_truth))
print("ground_truth shape:", ground_truth.shape)
print("ground_truth example:", ground_truth[:5])
