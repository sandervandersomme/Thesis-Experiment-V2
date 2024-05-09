hyperparams = {
    'learning_rate': 0.00005,  # Lower learning rates can be more stable
    'batch_size': 32,
    'noise_dim': 100,
    'hidden_dim': 128,
    'num_layers': 1,  # Number of layers in each module
    'critic_iterations': 5,  # Number of critic updates per generator update
    'clip_value': 0.01,
    # 'dropout_rate': 0.2,
    'embedding_dim': 16,  # Dimensionality of embedding space for generator
    'epochs': 10,
}

