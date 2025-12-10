# Archived Models

This folder contains all published models that are version controlled in the GitHub repository.

## Structure

```
archived_models/
├── legacy_salehi-2025-original/    # Salehi's original model from the paper
├── YYYY-MM-DD_model-name/          # Your archived models
│   ├── best_model.zip              # Best model checkpoint
│   ├── config.yaml                  # Training configuration
│   ├── info.txt                     # Experiment metadata
│   ├── progress.csv                 # Training progress (optional)
│   └── README.md                    # Model description and notes
└── ...
```

## Naming Convention

Folders are named with the date and a descriptive name:
- Format: `YYYY-MM-DD_model-name`
- Example: `2024-12-09_ppo-perlin-directional-seed10`

## What Gets Archived

For each archived model, we include:

1. **best_model.zip** - The best performing model checkpoint
2. **config.yaml** - Complete training configuration for reproducibility
3. **info.txt** - Experiment metadata (algorithm, seed, etc.)
4. **progress.csv** - Training progress logs (small file)
5. **README.md** - Description of the model, performance metrics, notes

## Archived Models

### Legacy Model (Salehi 2025)

The `legacy_salehi-2025-original/` folder contains the original model from:
- **Paper:** Salehi, Achkan. "Reinforcement Learning for Ballbot Navigation in Uneven Terrain." arXiv preprint arXiv:2505.18417 (2025)
- **Source:** Original implementation from the research paper

This model serves as a baseline for comparisons and understanding the original approach.

### Your Models

The following models have been archived from your training runs:

- **2025-12-04_ppo-flat-directional-seed10** - 10M steps, flat terrain (Final reward: 9.20 ± 0.00)
- **2025-12-03_ppo-perlin-directional-5.2M-steps** - 5.2M steps, perlin terrain
- **2025-12-04_ppo-flat-directional-1M-steps** - 1M steps, flat terrain
- **2025-12-03_ppo-perlin-directional-seed10** - Perlin terrain (Final reward: 2.53 ± 0.93)
- **2025-12-04_ppo-perlin-directional-seed10** - Perlin terrain model

Each archived model includes:
- **best_model.zip** - Best model checkpoint
- **config.yaml** - Complete training configuration
- **info.txt** - Experiment metadata
- **progress.csv** - Training progress logs
- **results/evaluations.npz** - Evaluation metrics over training (where available)
- **checkpoints/** - Key checkpoints (last, middle, first) for substantial training runs
- **README.md** - Model description with performance metrics

## Archiving Models

### Automatic Archiving

Use the scan and archive script to automatically archive good models:

```bash
# Scan and show what would be archived (dry run)
python scripts/utils/scan_and_archive_runs.py --dry-run --min-steps 100000

# Archive all models with >= 100k steps
python scripts/utils/scan_and_archive_runs.py --min-steps 100000

# Archive specific runs
python scripts/utils/scan_and_archive_runs.py --runs run1 run2 run3
```

### Manual Archiving

Use the archive script for individual models:

```bash
python scripts/utils/archive_model.py \
    --experiment outputs/experiments/runs/20241209_143022_ppo_perlin_directional_seed10 \
    --name "ppo-perlin-directional-seed10"
```

## Best Practices

1. **Only archive significant models** - Don't archive every run, only the best or most important ones
2. **Include metadata** - Always include config.yaml and info.txt for reproducibility
3. **Add descriptions** - Update README.md with:
   - Model performance metrics
   - Training details
   - Use cases
   - Known limitations
4. **Keep it small** - Only include essential files to keep repository size manageable
5. **Date format** - Use YYYY-MM-DD format for easy sorting

## Repository Organization

- **Training runs** (`outputs/experiments/runs/`) - All training experiments (git-ignored)
- **Archived models** (`outputs/experiments/archived_models/`) - Published models (version controlled)

This separation keeps the repository clean while preserving important models for reproducibility and sharing.

