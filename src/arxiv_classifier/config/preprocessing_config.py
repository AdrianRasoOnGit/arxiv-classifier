# src/arxiv_classifier/config/preprocessing_config.py

SPLITS_CONFIG = {
    # Get the 80/20 or 80/10/10
    "get_validation": True,

    # Proportions of the splits
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "batch_size": 50_000,

    # Seed for reproducibility
    "random_state": 24,
}
