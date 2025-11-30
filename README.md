# Visual Place Recognition Project

## Overview
This project implements various approaches for Visual Place Recognition (VPR) using deep learning techniques.

## Experimental Branches
This repository contains **nearly 20 experimental branches** exploring different architectures and approaches:

### Main Experimental Variants
- **CCC** - Core experimental implementation
- **CrossSalad1, CrossSalad2, crosssalad3** - Cross-attention based variants
- **PALSalad1** - Parameter-efficient learning approach
- **PETLSalad1** - Parameter-efficient transfer learning variant
- **mn_saladv21** - MobileNet-based variant

### Additional Experiments
- **add_resnet_spd** - ResNet with SPD-Conv integration
- **checkout** - Experimental checkout branch
- **easyModify** - Simplified modification approach
- And more...

To view all branches:
```bash
git branch -a
```

## Project Structure
```
├── dataloaders/          # Data loading utilities for various datasets
│   ├── GSVCitiesDataloader.py
│   ├── MapillaryDataset.py
│   ├── PittsburgDataset.py
│   └── val/             # Validation datasets
├── datasets/            # Dataset files and configurations
│   ├── msls_test/
│   ├── msls_val/
│   ├── Nordland/
│   ├── Pittsburgh/
│   └── SPED/
├── models/              # Model architectures
│   ├── aggregators/
│   └── backbones/
├── utils/               # Utility functions
│   ├── losses.py
│   └── validation.py
├── main.py             # Main training script
├── eval.py             # Evaluation script
└── vpr_model.py        # VPR model implementation
```

## Installation
```bash
conda env create -f environment.yml
conda activate <env_name>
```

## Usage

### Training
```bash
python main.py
```

### Evaluation
```bash
python eval.py
```

## Datasets
The project supports multiple VPR datasets:
- GSV-Cities
- Mapillary Street-Level Sequences (MSLS)
- Pittsburgh
- Nordland
- SPED

## License
See [LICENSE](LICENSE) file for details.
