import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from vpr_model import VPRModel
from utils.validation import get_validation_recalls
# Dataloader
import faiss
import faiss.contrib.torch_utils
from prettytable import PrettyTable

from dataloaders.val.NordlandDataset import NordlandDataset
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import os
# from dataloaders.val.MapillaryDataset import MSLS
# from dataloaders.val.MapillaryTestDataset import MSLSTest
# from dataloaders.val.PittsburghDataset import PittsburghDataset
# from dataloaders.val.SPEDDataset import SPEDDataset

# VAL_DATASETS = ['MSLS', 'MSLS_Test', 'pitts30k_test', 'pitts250k_test', 'Nordland', 'SPED']
VAL_DATASETS = ['Nordland']


def input_transform(image_size=None):
    MEAN=[0.485, 0.456, 0.406]; STD=[0.229, 0.224, 0.225]
    if image_size:
        return T.Compose([
            T.Resize(image_size,  interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
    else:
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])

def get_val_dataset(dataset_name, image_size=None):
    dataset_name = dataset_name.lower()
    transform = input_transform(image_size=image_size)
    
    if 'nordland' in dataset_name:    
        ds = NordlandDataset(input_transform=transform)
    # elif 'msls_test' in dataset_name:
    #     ds = MSLSTest(input_transform=transform)

    # elif 'msls' in dataset_name:
    #     ds = MSLS(input_transform=transform)

    # elif 'pitts' in dataset_name:
    #     ds = PittsburghDataset(which_ds=dataset_name, input_transform=transform)

    # elif 'sped' in dataset_name:
    #     ds = SPEDDataset(input_transform=transform)
    else:
        raise ValueError
    
    num_references = ds.num_references
    num_queries = ds.num_queries
    ground_truth = ds.ground_truth
    # print( "==================  label of some place ==================== ")  
    # print( ground_truth[:5][1] ) 
    return ds, num_references, num_queries, ground_truth

def get_descriptors(model, dataloader, device):
    descriptors = []
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for batch in tqdm(dataloader, 'Calculating descritptors...'):
                imgs, labels = batch
                output = model(imgs.to(device)).cpu()
                descriptors.append(output)

    return torch.cat(descriptors)

def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Giúp khôi phục ảnh từ dạng normalized về dạng hiển thị được (0-1 range).
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # t = t * std + mean
    return torch.clamp(tensor, 0, 1)
    
def load_model(ckpt_path):
    model = VPRModel(
        backbone_arch='dinov2_vitb14',
        backbone_config={
            'num_trainable_blocks': 4,
            'return_token': True,
            'norm_layer': True,
        },
        agg_arch='SALAD',
        agg_config={
            'num_channels': 768,
            'num_clusters': 64,
            'cluster_dim': 128,
            'token_dim': 256,
        },
    )

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # Lấy đúng phần state_dict bên trong
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    model = model.eval()
    model = model.to('cuda')
    print(f"Loaded model from {ckpt_path} successfully!")
    return model

def parse_args():
    parser = argparse.ArgumentParser(
        description="Eval VPR model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Model parameters
    parser.add_argument("--ckpt_path", type=str, required=True, default=None, help="Path to the checkpoint")
    
    # Datasets parameters
    parser.add_argument(
        '--val_datasets',
        nargs='+',
        default=VAL_DATASETS,
        help='Validation datasets to use',
        choices=VAL_DATASETS,
    )
    parser.add_argument('--image_size', nargs='*', default=None, help='Image size (int, tuple or None)')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')

    args = parser.parse_args()

    # Parse image size
    if args.image_size:
        if len(args.image_size) == 1:
            args.image_size = (args.image_size[0], args.image_size[0])
        elif len(args.image_size) == 2:
            args.image_size = tuple(args.image_size)
        else:
            raise ValueError('Invalid image size, must be int, tuple or None')
        
        args.image_size = tuple(map(int, args.image_size))

    return args


if __name__ == '__main__':

    torch.backends.cudnn.benchmark = True

    args = parse_args()
    
    model = load_model(args.ckpt_path)

    for val_name in args.val_datasets:
        print( type(val_name)) 
        val_dataset, num_references, num_queries, ground_truth = get_val_dataset(val_name, args.image_size)
        print( "iamge size : " , args.image_size) 
        val_loader = DataLoader(val_dataset, num_workers=16, batch_size=args.batch_size, shuffle=False, pin_memory=True)

        print(f'Evaluating on {val_name}')
        descriptors = get_descriptors(model, val_loader, 'cuda')
        
        print(f'Descriptor dimension {descriptors.shape[1]}')
        r_list = descriptors[ : num_references]
        q_list = descriptors[num_references : ]

        print('total_size', descriptors.shape[0], num_queries + num_references) 

        # descriptors dimension is 8448 
        # testing = True # isinstance(val_dataset, MSLSTest)

        preds = get_validation_recalls(
            r_list=r_list,
            q_list=q_list,
            k_values=[1, 5, 10, 15, 20, 25],
            gt=ground_truth,
            print_results=True,
            dataset_name=val_name,
            faiss_gpu=False,
            testing=False,
        )

        # Chọn 2 query để hiển thị
        
        for q_idx in [1000, 1, 2020, 40, 60, 100, 300, 700, 800, 14, 50 ]:
            print(f"\nQuery index: {q_idx}")
            pred_ids = preds[q_idx][:7]  # top-4 predictions
        
            # Lấy ảnh query và reference từ dataset (tensor), rồi denormalize để hiển thị
            query_img, _ = val_dataset[num_references + q_idx]
            query_img = denormalize(query_img.clone())
            query_img_np = TF.to_pil_image(query_img)
        
            fig, axes = plt.subplots(1, 8, figsize=(15, 3))
            axes[0].imshow(query_img_np)
            axes[0].set_title("Query")
            axes[0].axis('off')
        
            for i, pred_id in enumerate(pred_ids):
                ref_img, _ = val_dataset[pred_id]
                ref_img = denormalize(ref_img.clone())
                ref_img_np = TF.to_pil_image(ref_img)
        
                axes[i+1].imshow(ref_img_np)
                axes[i+1].set_title(f"Top-{i+1}")
                axes[i+1].axis('off')
        
            plt.tight_layout()
            plt.savefig(f"./query_{q_idx}_top4.png")
            plt.close()


        # print( preds ) 
        
        # if testing:
        #     val_dataset.save_predictions(preds, '../preds.txt')

        del descriptors
        print('========> DONE!\n\n')

