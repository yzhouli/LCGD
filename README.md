# LCGD: Langevin-Inspired Controlled Graph Dynamics for Robust Cascades Prediction

## Model Overview

LCGD replaces the linear GNN encoder in cascade prediction models with a **Controlled Graph Dynamics (CGD)**, a Langevin-inspired  energy-based graph dynamics encoder.

### Architecture

1. **CGD Encoder** (`layers/CGD.py`): Iterative controlled dynamics over the cascade-user bipartite graph
2. **Sequence Encoder**: TransformerBlock attention over user sequences with positional encoding
3. **SSL Layer Loss**: Contrastive loss between consecutive CDAT iterations (layer-wise)
4. **Prediction Head**: Linear projection to user vocabulary

### Key Hyperparameters (6 swept)

| Parameter | Flag | Role |
|-----------|------|------|
| η (step size) | `--cdat_alpha` | Dynamics iteration step size |
| r (rank) | `--cdat_rank` | Low-rank attractor dimension |
| h (heads) | `--cdat_heads` | Graph attention heads |
| τ (temperature) | `--ssl_temp` | SSL contrastive temperature |
| λ (reg) | `--ssl_reg` | SSL loss weight |
| α (SSL balance) | `--alpha` | User/item SSL branch balance |

### Fixed Parameters

| Parameter | Value | Flag |
|-----------|-------|------|
| L (layers) | 2 | `--cdat_layers` |
| σ (noise) | 0.02 | `--cdat_noise_std` |
| c (inhibition) | learned | (model parameter) |

## File Structure

```
LCGD/
├── run.py                  # Main entry point
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── models/
│   └── LCGD.py             # LCGD model
├── layers/
│   ├── CGD.py  # CGD encoder (controlled dynamics core)
│   ├── GraphEncoders.py     # Alternative GNN encoders (ablation)
│   └── TransformerBlock.py  # Transformer attention block
├── helpers/
│   ├── BaseLoader.py        # Data loading, train/valid/test split
│   └── BaseRunner.py        # Training/evaluation loop
├── utils/
│   ├── Constants.py         # PAD constant, etc.
│   ├── Metrics.py           # HITS, MAP, NDCG computation
│   ├── Utils.py             # Logging, seeding, model info
│   └── Optim.py             # Optimizer utilities
└── data/
    ├── douban/
    │   ├── cascades.txt     # Cascade sequences
    │   └── edges.txt        # Social graph edges
    ├── android/
    │   ├── cascades.txt     # Cascade sequences
    │   └── edges.txt        # Social graph edges
    ├── christianity/
    │   ├── cascades.txt     # Cascade sequences
    │   └── edges.txt        # Social graph edges
    └── memetracker/
        ├── cascades.txt     # Cascade sequences
        └── edges.txt        # Social graph edges
```

## Environment

```
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
torch-geometric>=2.6.0
tqdm>=4.65.0
pandas>=2.0.0
```

## How to Run

### Full training (200 epochs, default params, select GPU)

```bash
python run.py --model_name LCGD -data_name douban -epoch 200 -gpu 0
```

### Custom hyperparameters

```bash
python run.py --model_name LCGD -data_name douban \
    -epoch 200 --cgd_alpha 0.3 \
    --cgd_rank 4 --cgd_heads 4 \
    --ssl_temp 0.9 --ssl_reg 1e-4 --alpha 1e-3
```

### Available datasets
`douban`, `android`, `christianity`, `memetracker`

### Output
- Log: `log/LCGD/LCGD__<dataset>.txt`
- Model checkpoint: `saved/LCGD/LCGD__<dataset>.pt`
- Metrics printed: HITS@10/20/50/100, MAP@10/20/50/100, NDCG@10/20/50/100
