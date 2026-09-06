# Potato Leaf Disease Classification

Hybrid ConvNeXt + Transformer classifier for potato leaf disease, built in
TensorFlow/Keras.

## Overview

Classifies potato leaf images into three classes — Early Blight, Late Blight,
and Healthy. The architecture pairs a pretrained CNN backbone for local texture
and lesion detail with a self-attention block that lets the model relate
features across spatially distant regions of the leaf.

## Architecture

| Component | Detail |
|---|---|
| Backbone | ConvNeXt-Tiny, ImageNet-pretrained, **frozen** (`trainable=False`) |
| Feature map | 7 × 7 × 768, reshaped to a 49-token sequence |
| Attention | 1 multi-head self-attention block, 4 heads, `key_dim=256`, residual connections + LayerNorm |
| Feed-forward | Dense(256, ReLU) → Dense(768), residual |
| Head | GlobalAveragePooling1D → Dropout(0.3) → Dense(128, ReLU) → Dense(3, softmax) |
| Parameters | 31.5M total, **3.6M trainable** (27.8M frozen in the backbone) |

The backbone is frozen throughout — there is no fine-tuning stage. All learning
happens in the attention block and MLP head.

## Data

[PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease), potato subset.

| Split | Images |
|---|---|
| Train | 5,403 |
| Validation | 763 |

Input resolution 224 × 224, batch size 32. Images are passed in the [0, 255]
range, since Keras ConvNeXt applies its own normalization internally.

## Results

Trained for 20 epochs with Adam (default LR) and categorical cross-entropy.

| Metric | Value |
|---|---|
| Validation accuracy (`model.evaluate`, final model) | **97.9%** |
| Validation loss | 0.20 |
| Final-epoch training accuracy | 99.6% |

Evaluated with a per-class confusion matrix, precision/recall/F1 via
`classification_report`, and per-class ROC-AUC — not accuracy alone.

> Note on which number to quote: the final training epoch reported 97.4%
> validation accuracy, `model.evaluate()` on the same set returned 97.9%, and
> the best epoch reached 98.2%. The 97.9% figure above is the saved final model
> evaluated explicitly. There is currently no separate test split.

## Known limitations

Recorded honestly rather than omitted:

- **No held-out test set.** Only train and validation splits exist. The validation set was never trained on, but it was scored every epoch and used to select the reported figure, so it is not a clean held-out estimate.
- **No ablation.** The attention block adds ~3.1M parameters, but the model was
  never trained without it, so its contribution is unmeasured. A frozen backbone
  with a plain pooling head is the missing baseline.
- **Overfitting is not controlled.** Validation loss rises from 0.077 (epoch 3)
  to 0.386 (epoch 13) while training accuracy climbs toward 0.997. There is no
  early stopping, LR schedule, or best-checkpoint saving — the final epoch is
  kept regardless of whether it was the best.
- **No augmentation.** `keras_cv` is imported but unused.
- **PlantVillage is a lab-condition dataset.** Single leaves on uniform
  backgrounds. High accuracy here is expected and is known not to transfer to
  field photographs with natural lighting, occlusion, and cluttered backgrounds.

## Repository

- `ConvNeXtTiny(CNN)+Transformer+MLP_Classifier_pynb.ipynb` — full pipeline: data
  loading, model definition, training, evaluation, and inference.

## Running it

```bash
pip install tensorflow keras-cv
```

Open the notebook in Google Colab with a GPU runtime. The dataset is expected at
`DATA_DIR` with this structure:

```
PlantVillage/
├── train/
│   ├── Early_Blight/
│   ├── Late_Blight/
│   └── Healthy/
└── val/
    ├── Early_Blight/
    ├── Late_Blight/
    └── Healthy/
```

## Next steps

- Ablate the attention block against a pooling-only baseline
- Add a held-out test split and report test accuracy
- Add `EarlyStopping` and `ModelCheckpoint` on validation loss
- Unfreeze the top ConvNeXt stages and fine-tune at a low learning rate
- Evaluate on field-condition images to measure the domain gap
