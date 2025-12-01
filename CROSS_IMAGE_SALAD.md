# Cross-Image SALAD

Mở rộng SALAD (Sinkhorn Algorithm for Locally Aggregated Descriptors) với Cross-Image Learning để khai thác mối quan hệ giữa các ảnh cùng một place.

## Ý tưởng

Trong Visual Place Recognition, mỗi place thường có nhiều ảnh từ các góc nhìn/thời điểm khác nhau. Cross-Image SALAD sử dụng Transformer Encoder để cho phép các ảnh cùng place "chia sẻ" thông tin với nhau, tạo ra descriptor phong phú hơn.

## Cấu trúc

```
models/aggregators/
├── salad_base.py           # SALADBase - SALAD gốc (single-image)
├── cross_image_encoder.py  # CrossImageEncoder - Transformer cho cross-image
└── salad.py                # SALAD - kết hợp cả 2
```

### SALADBase (`salad_base.py`)
- SALAD gốc, xử lý từng ảnh độc lập
- Có thể load pretrained weights
- Dùng cho query inference

### CrossImageEncoder (`cross_image_encoder.py`)
- Transformer Encoder học mối quan hệ giữa các ảnh cùng place
- Input: cluster features từ `img_per_place` ảnh
- Output: enhanced features với residual connection

### SALAD (`salad.py`)
- Kết hợp SALADBase + CrossImageEncoder
- Hỗ trợ load pretrained weights riêng cho từng phần
- 3 forward modes:
  - `forward()`: Auto cross-image khi training
  - `forward_single()`: Single image (query)
  - `forward_database()`: Database với cross-image

## Cách sử dụng

### Load pretrained weights
```python
from models.aggregators import SALAD

model = SALAD(
    num_channels=768,
    num_clusters=64,
    cluster_dim=128,
    token_dim=256,
    img_per_place=4,
)

# Load pretrained SALADBase
model.load_base_weights('pretrainedWeight/Salad/last.ckpt')

# Optional: freeze base, chỉ train cross-image
model.freeze_base()
```

### Training
```bash
# Windows
train.bat

# Linux/Mac
chmod +x train.sh
./train.sh

# Hoặc trực tiếp
python train_cross_image.py --pretrained_path pretrainedWeight/Salad/last.ckpt --freeze_base
```

### Inference
```python
# Query (single image)
query_desc = model.forward_single(x)

# Database (với cross-image, batch phải chia hết cho img_per_place)
db_desc = model.forward_database(x)
```

## Training Options

| Option | Command | Mô tả |
|--------|---------|-------|
| Freeze base | `--freeze_base` | Chỉ train CrossImageEncoder |
| Fine-tune | không có flag | Train toàn bộ model |
| From scratch | `--pretrained_path ""` | Không load pretrained |

## Lưu ý

1. **Batch size**: Phải là bội số của `img_per_place` (default=4)
2. **Validation**: Tự động dùng single-image mode (không cross-image)
3. **Database embedding**: Cần group ảnh theo place khi dùng `forward_database()`

## Files

| File | Mô tả |
|------|-------|
| `models/aggregators/salad_base.py` | SALADBase module |
| `models/aggregators/cross_image_encoder.py` | CrossImageEncoder module |
| `models/aggregators/salad.py` | Combined SALAD |
| `train_cross_image.py` | Training script với argparse |
| `train.sh` | Shell script (Linux/Mac) |
| `train.bat` | Batch script (Windows) |
