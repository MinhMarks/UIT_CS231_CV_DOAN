"""
Evaluate pretrained SALAD with ORIGINAL architecture (no cross-image).
This establishes the baseline performance.
"""
import argparse
import torch
import pytorch_lightning as pl

from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule
from models.aggregators.salad_original import SALADOriginal
from models import helper
import utils


class VPRModelOriginal(pl.LightningModule):
    """VPR Model with original SALAD (no cross-image)."""

    def __init__(self, backbone_arch="dinov2_vitb14", backbone_config=None, agg_config=None, faiss_gpu=False):
        super().__init__()

        if backbone_config is None:
            backbone_config = {
                "num_trainable_blocks": 4,
                "return_token": True,
                "norm_layer": True,
            }

        if agg_config is None:
            agg_config = {
                "num_channels": 768,
                "num_clusters": 64,
                "cluster_dim": 128,
                "token_dim": 256,
            }

        self.backbone = helper.get_backbone(backbone_arch, backbone_config)
        self.aggregator = SALADOriginal(**agg_config)
        self.faiss_gpu = faiss_gpu
        self.val_outputs = {}

    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        return x

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        places, _ = batch
        descriptors = self(places)

        if dataloader_idx is None:
            dataloader_idx = 0

        if dataloader_idx not in self.val_outputs:
            self.val_outputs[dataloader_idx] = []

        self.val_outputs[dataloader_idx].append(descriptors.detach().cpu())
        return descriptors.detach().cpu()

    def on_validation_epoch_start(self):
        self.val_outputs = {}

    def on_validation_epoch_end(self):
        dm = self.trainer.datamodule
        num_val_datasets = len(dm.val_datasets) if hasattr(dm, "val_datasets") else 1

        for i in range(num_val_datasets):
            if i not in self.val_outputs:
                continue

            feats = torch.cat(self.val_outputs[i], dim=0)

            val_set_name = dm.val_set_names[i]
            val_dataset = dm.val_datasets[i]

            if "pitts" in val_set_name.lower():
                num_references = val_dataset.dbStruct.numDb
                positives = val_dataset.getPositives()
            elif "msls" in val_set_name.lower():
                num_references = val_dataset.num_references
                positives = val_dataset.pIdx
            else:
                continue

            r_list = feats[:num_references]
            q_list = feats[num_references:]

            recalls = utils.get_validation_recalls(
                r_list=r_list,
                q_list=q_list,
                k_values=[1, 5, 10, 15, 20, 50, 100],
                gt=positives,
                print_results=True,
                dataset_name=val_set_name,
                faiss_gpu=self.faiss_gpu,
            )

            self.log(f"{val_set_name}/R1", recalls[1], prog_bar=False, logger=True)
            self.log(f"{val_set_name}/R5", recalls[5], prog_bar=False, logger=True)
            self.log(f"{val_set_name}/R10", recalls[10], prog_bar=False, logger=True)

        self.val_outputs = {}


def load_pretrained_weights(model, checkpoint_path):
    """Load pretrained weights into original SALAD model."""
    print(f"\nLoading weights from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Map weights
    new_state_dict = {}

    for k, v in state_dict.items():
        new_key = k

        # Handle different checkpoint formats
        if k.startswith("model."):
            new_key = k[6:]  # Remove "model."

        new_state_dict[new_key] = v

    # Load with strict=False to handle any mismatches
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

    print(f"Loaded {len(new_state_dict) - len(missing)} weights")
    if missing:
        print(f"Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    return model


def main(args):
    print("\n" + "=" * 60)
    print(" Evaluating ORIGINAL SALAD (Baseline)")
    print("=" * 60)

    # DataModule
    datamodule = GSVCitiesDataModule(
        batch_size=args.batch_size,
        img_per_place=4,
        min_img_per_place=4,
        shuffle_all=False,
        random_sample_from_each_place=True,
        image_size=(args.image_size, args.image_size),
        num_workers=args.num_workers,
        show_data_stats=True,
        val_set_names=args.val_sets.split(","),
    )

    # Model
    model = VPRModelOriginal(
        backbone_arch="dinov2_vitb14",
        backbone_config={
            "num_trainable_blocks": 4,
            "return_token": True,
            "norm_layer": True,
        },
        agg_config={
            "num_channels": 768,
            "num_clusters": 64,
            "cluster_dim": 128,
            "token_dim": 256,
        },
        faiss_gpu=args.faiss_gpu,
    )

    # Load pretrained weights
    model = load_pretrained_weights(model, args.checkpoint)

    # Setup datamodule
    datamodule.setup(stage="fit")

    # Trainer
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        logger=False,
    )

    # Evaluate
    print("\n" + "=" * 60)
    print(" Running Evaluation")
    print("=" * 60 + "\n")

    trainer.validate(model=model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Original SALAD Baseline")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="pretrainedWeight/Salad/last.ckpt",
        help="Path to pretrained checkpoint",
    )
    parser.add_argument(
        "--val_sets",
        type=str,
        default="msls_val",
        help="Comma-separated validation sets",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--faiss_gpu", action="store_true")

    args = parser.parse_args()
    main(args)
