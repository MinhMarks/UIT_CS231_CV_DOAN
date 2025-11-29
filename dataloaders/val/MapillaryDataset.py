from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image

# Get project root directory (2 levels up from this file)
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))

DATASET_ROOT = os.path.join(_project_root, 'datasets') + '/'
GT_ROOT = os.path.join(_project_root, 'datasets', 'msls_val') + '/'

class MSLS(Dataset):
    def __init__(self, input_transform = None):
        

        self.input_transform = input_transform

        self.dbImages = np.load(os.path.join(GT_ROOT, 'msls_val_dbImages.npy'))
        self.qIdx = np.load(os.path.join(GT_ROOT, 'msls_val_qIdx.npy'))
        self.qImages = np.load(os.path.join(GT_ROOT, 'msls_val_qImages.npy'))
        self.ground_truth = np.load(os.path.join(GT_ROOT, 'msls_val_pIdx.npy'), allow_pickle=True)
        
        print(self.dbImages) 
        # reference images then query images
        self.images = np.concatenate((self.dbImages, self.qImages[self.qIdx]))
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages[self.qIdx])
    
    def __getitem__(self, index):
        img = Image.open(DATASET_ROOT + self.images[index])

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)