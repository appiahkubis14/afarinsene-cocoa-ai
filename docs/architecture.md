# Afarinsene — Architecture Notes

## CNN Design Rationale

### Why a custom architecture instead of transfer learning?

| Concern | Explanation |
|---|---|
| **Dataset size** | 476 images — fine-tuning a ResNet-50 risks overfitting the last few layers. |
| **Domain specificity** | ImageNet features (dogs, cats, objects) are weakly relevant to cocoa leaf disease textures. |
| **Speed** | A 421k-param model runs in 23 ms on CPU vs 45–62 ms for VGG-16/ResNet-50. |
| **Edge deployment** | Small model → small TFLite file (≈1.7 MB quantised) → fits on low-memory Android devices. |

### Block structure

```
Block i (filters F):
  Conv2D(F, 3×3) + BN + ReLU
  Conv2D(F, 3×3) + BN + ReLU
  MaxPool(2×2)
  Dropout(0.25)
```

Four blocks with F ∈ {32, 64, 128, 256}. After block 4, the spatial
dimension is 256/16 = 16 × 16 (for 256×256 input).

### Why GlobalAveragePooling instead of Flatten?

`Flatten` at 256 filters × 16×16 = 65,536 features → dense layer of
65,536 × 256 = 16.7 M parameters — huge overfitting risk for 476 images.

`GlobalAveragePooling2D` collapses spatial dimensions to a single 256-dim
vector regardless of input resolution. Dense layers see only 256 features.

### Regularisation

* **BatchNorm** after every Conv2D — stabilises training, acts as mild regulariser.
* **Dropout 0.25** after each max-pool block.
* **Dropout 0.5 / 0.3** in dense layers.
* **L2(0.0001)** on dense layer kernels.
* **Gradient clipping** (`clipnorm=1.0`) prevents exploding gradients during
  LR increases after ReduceLROnPlateau reductions.

---

## Data Pipeline

```
Raw images (uint8, disk)
  │
  ▼ DatasetLoader
uint8 arrays in RAM

  │
  ▼ Preprocessor
float32, [0,1]
Stratified 70/15/15 split

  │ (train set only)
  ▼ Augmenter
Class-balanced set
MixUp applied
```

### Why balance only the training set?

Validation and test sets must reflect the **true class distribution** of
real-world usage. Balancing them would produce an optimistic evaluation.
Training set balancing via augmentation prevents the model from ignoring
minority classes (e.g., Black Pod at 26% of data).

---

## Checkpoint Strategy

Two files are maintained:

| File | Saved when | Purpose |
|---|---|---|
| `best_model.keras` | `val_accuracy` improves | Deployment artifact |
| `latest_checkpoint.keras` | Every epoch | Enable `--resume` |

The `.keras` format (vs `.h5`) stores the full model graph + weights +
optimizer state, enabling seamless resume.

---

## TFLite Export

| Format | Ops | Size | Use case |
|---|---|---|---|
| Float32 | Full precision | ~6.5 MB | Android/iOS high-accuracy |
| INT8 | Quantised (default+INT8) | ~1.7 MB | Edge, Raspberry Pi, low RAM |

INT8 calibration uses 200 training samples fed through a representative
dataset generator. Input/output types are kept as `float32` for
compatibility with standard inference pipelines.
