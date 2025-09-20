from torch.utils.data import Dataset

import pandas as pd
import numpy as np
from PIL import Image
import os

DATASET_ROOT = '/kaggle/input/msls-dataset/'
GT_ROOT = '/kaggle/input/salad/pytorch/mn_saladv21/1/datasets/msls_val/'

class MSLS(Dataset):
    def __init__(self, input_transform = None):
        

        self.input_transform = input_transform

        self.dbImages = np.load(GT_ROOT+'msls_val/msls_val_dbImages.npy')
        self.qIdx = np.load(GT_ROOT+'msls_val/msls_val_qIdx.npy')
        self.qImages = np.load(GT_ROOT+'msls_val/msls_val_qImages.npy')
        self.ground_truth = np.load(GT_ROOT+'msls_val/msls_val_pIdx.npy', allow_pickle=True)


        # path example: train_val/cph/database/images/HU9GEfLAB9pm5RmjW4MLhg.jpg
        folder = os.path.dirname(os.path.dirname(self.dbImages[0]))  # train_val/cph/database
        csv_path = os.path.join(DATASET_ROOT, folder, "postprocessed.csv" ) 
        
        meta = pd.read_csv(csv_path)

        filtered_db = []
        for path in self.dbImages:

            img_name = os.path.basename(path)
            if img_name in meta["key"].values:
                if meta.loc[meta["key"] == img_name, "night"].values[0] == 0:  # day only
                    filtered_db.append(path)
    
        self.dbImages = np.array(filtered_db)

        filtered_q = []
        filtered_qIdx = []

        path = self.qImages[0]
        folder = os.path.dirname(os.path.dirname(path))  # train_val/cph/queries
        csv_path = os.path.join(DATASET_ROOT, folder, "postprocessed.csv")

        meta = pd.read_csv(csv_path)

        for i, idx in enumerate(self.qIdx):

            path = self.qImages[idx]
            img_name = os.path.basename(path)
            if img_name in meta["key"].values:
                if meta.loc[meta["key"] == img_name, "night"].values[0] == 1:  # night only
                    filtered_q.append(path)
                    filtered_qIdx.append(idx)
                    
        self.qImages = np.array(filtered_q)
        self.qIdx = np.array(filtered_qIdx)

        # reference images then query images
        self.images = np.concatenate((self.dbImages, self.qImages[self.qIdx]))
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages[self.qIdx])
    
    def __getitem__(self, index):
        
        img_path = os.path.join(DATASET_ROOT, self.images[index])
        img = Image.open(img_path)

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)