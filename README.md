# Prompted Segmentation for Drywall QA

Text-conditioned segmentation for drywall quality assurance — detecting **cracks** and **taping areas** using natural-language prompts.

## Project Structure

```
drywall_segmentation/
├── datasets/
│   ├── cracks/          # Crack segmentation dataset (train/valid/test)
│   └── taping/          # Taping area segmentation dataset (train/valid/test)
├── models/              # Saved model checkpoints
├── outputs/             # Prediction masks (PNG)
├── scripts/
│   ├── dataset.py       # Dataset loader (COCO → binary masks + prompts)
│   ├── train.py         # CLIPSeg fine-tuning script
│   ├── inference.py     # Generate prediction masks
│   └── metrics.py       # Compute mIoU, Dice, Precision, Recall
├── report/              # Evaluation results & visual examples
│   └── report.pdf
├── README.md
└── requirements.txt
```

## Approach

**Model**: [CLIPSeg](https://huggingface.co/CIDAS/clipseg-rd64-refined) — a text-conditioned segmentation model built on CLIP.

- **Encoder**: Frozen CLIP ViT-B/16 (pretrained, not fine-tuned)
- **Decoder**: Fine-tuned lightweight decoder that maps CLIP features → segmentation mask
- **Loss**: BCE + Dice combined loss (0.5 weight each)
- **Prompts**: Multiple prompt variants per class for robustness

| Dataset | Prompt Examples | Train | Val | Test |
|---------|----------------|-------|-----|------|
| Cracks | `"segment crack"`, `"segment wall crack"` | 3758 | 805 | 806 |
| Taping | `"segment taping area"`, `"segment drywall seam"` | 715 | 153 | 154 |

## Setup

```bash
pip install -r requirements.txt
```

**Seeds**: Random seed = `42` for all data splits and model training.

## Training

```bash
# Train on both datasets jointly
python scripts/train.py --dataset both --epochs 30 --batch_size 4 --lr 1e-4 --seed 42

# Or train on individual datasets
python scripts/train.py --dataset cracks --epochs 30 --batch_size 4 --lr 1e-4
python scripts/train.py --dataset taping --epochs 50 --batch_size 4 --lr 1e-4
```

Checkpoints are saved to `models/`.

## Inference

```bash
# Generate prediction masks
python scripts/inference.py --dataset both --checkpoint models/clipseg_both_best.pth

# With visual comparison grids for report
python scripts/inference.py --dataset both --checkpoint models/clipseg_both_best.pth --visuals
```

Output masks are saved as single-channel PNG files with values `{0, 255}` to `outputs/`.
Filenames follow the format: `{image_id}__segment_crack.png`.

## Evaluation

```bash
python scripts/metrics.py --dataset both --checkpoint models/clipseg_both_best.pth
```

Metrics computed: **mIoU**, **Dice**, **Precision**, **Recall**, **Pixel Accuracy**.

## Prediction Mask Format

- **Format**: PNG, single-channel, same spatial size as source image
- **Values**: `{0, 255}` (background / foreground)
- **Naming**: `{image_id}__{prompt_slug}.png` (e.g., `123__segment_crack.png`)

## Runtime & Footprint

See `outputs/inference_stats.json` and `models/training_summary_*.json` after running the pipeline.

> [!NOTE]
> **GPU Training Notice**: If you encounter a CUDA error like `The NVIDIA driver on your system is too old` (e.g. Driver 535 / CUDA 12.2), you should install the PyTorch version compiled for CUDA 12.1 using:
> `pip install --break-system-packages torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
