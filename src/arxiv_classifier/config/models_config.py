# src/arxiv_classifier/config/models_config.py

# Architecture settings

FMLP_CONFIG = {
    # Hidden layers
    "hidden_layers": 2,
    "hidden_dim": 256, # choose between 128, 256, or 512 if you dare

    # Optimization
    "learning_rate": 0.0005,
    "dropout_rate": 0.4, # before 0.7

    # Regularization
    "lam": 0.01, # before 0.1

    # Training
    "epochs": 50,
    "batch_size": 2048,

    # Some further performance settings:
    "max_train": 400000,
    "max_test": 5000,
}

FMLP_PROFILES_CONFIG = {
    "tfidf": {
    },

    "ngram": {
        "hidden_layers": 2,
        "hidden_dim": 512,
        "dropout_rate": 0.55,
        "lam": 0.03,
        "learning_rate": 0.00065,
        "epochs": 70,
    },
    
    "embeddings": {
        "hidden_layers": 2,
        "hidden_dim": 192,
        "dropout_rate": 0.3,
        "learning_rate": 0.0007,
        "epochs": 80,
    },
}

    
