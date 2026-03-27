# 🧱 Drywall Defect Segmentation using CLIPSeg

## 🚀 Overview

This project presents a **prompt-based segmentation system** for detecting drywall defects using **CLIPSeg**, a vision-language model.

The system identifies:

* 🔹 **Cracks**
* 🔹 **Taping seams (drywall joints)**

Unlike traditional segmentation models, this approach leverages **natural language prompts** to guide segmentation, making it flexible and adaptable.

---

## 🎯 Problem Statement

Drywall inspection is critical in construction quality assurance. However:

* Cracks vary in shape and size
* Taping seams are **thin structures**, making them difficult to detect
* Traditional models struggle with **class imbalance and fine details**

---

## 💡 Key Contributions

### ✅ Prompt-Based Segmentation

* Used **CLIPSeg (`CIDAS/clipseg-rd64-refined`)**
* Enabled segmentation via prompts like:

  * `"segment crack"`
  * `"segment taping area"`

---

### ✅ Thin Structure Handling (Major Challenge Solved)

Taping seams initially failed due to:

* Extreme thinness (1–2 pixels after resizing)
* Class imbalance (dominant background)

✔ Solutions implemented:

* Increased resolution (**512×512**)
* Dice-weighted loss (focus on small regions)
* **Mask dilation** to enhance thin features
* Full model fine-tuning (`--train_all`)

---

### ✅ Multi-Task Learning

* Combined datasets for:

  * Cracks (large, irregular defects)
  * Taping (thin linear defects)
* Improved generalization across defect types

---

## 📊 Results

### 🔹 Cracks

* **mIoU:** 0.53
* **mDice:** 0.67

### 🔹 Taping (after fixes)

* Initially: ❌ 0.00 Dice (model collapse)
* After improvements: ✅ Significant recovery (thin seam detection enabled)

---

## 🖼️ Visual Results

| Input            | Ground Truth      | Prediction      |
| ---------------- | ----------------- | --------------- |
| ✔️ Drywall image | ✔️ Annotated mask | ✔️ Model output |

📁 See:

```
report/visual_examples_cracks.png
report/visual_examples_taping.png
```

---

## ⚙️ Project Structure

```
drywall_segmentation/
│
├── scripts/              # Training, inference, metrics
├── report/               # Results & visualizations
├── models/               # Saved checkpoints
├── datasets/             # (ignored in Git)
├── masks/                # Generated masks (ignored)
├── outputs/              # Predictions (ignored)
├── README.md
└── requirements.txt
```

---

## 🧪 Installation

```bash
git clone https://github.com/YOUR_USERNAME/drywall-segmentation.git
cd drywall-segmentation

pip install -r requirements.txt
```

---

## 🏋️ Training

```bash
python scripts/train.py \
--dataset both \
--epochs 40 \
--batch_size 8 \
--image_size 512 \
--train_all
```

---

## 🔍 Inference

```bash
python scripts/inference.py \
--dataset both \
--checkpoint models/clipseg_both_best.pth \
--image_size 512 \
--visuals
```

---

## 📈 Evaluation

```bash
python scripts/metrics.py \
--dataset both \
--checkpoint models/clipseg_both_best.pth
```

---

## 🧠 Key Insights

* Vision-language models can **generalize segmentation tasks via prompts**
* Thin structures require:

  * Higher resolution
  * Specialized loss functions
  * Morphological preprocessing
* Combining datasets improves robustness

---

## ⚡ Performance

* ⚙️ Model: CLIPSeg (ViT-based)
* ⏱️ Inference time: ~130 ms/image (GPU)
* 💾 Model size: ~575 MB

---

## 🔮 Future Work

* Improve taping detection using:

  * Edge-aware losses
  * Multi-scale feature learning
* Deploy as real-time inspection tool
* Extend to additional construction defects

---

## 🤝 Acknowledgements

* HuggingFace Transformers
* CLIPSeg by CIDAS
* Roboflow datasets

---

## 📬 Contact

If you're a recruiter or collaborator, feel free to connect!

* 💼 Focus: AI/ML, Computer Vision, Deep Learning
* 🛠️ Skills: Python, PyTorch, OpenCV, Transformers

---

⭐ If you found this useful, consider starring the repo!
