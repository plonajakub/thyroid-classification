class Parameters:
    def __init__(self):
        self.learning_rate = 'adaptive'
        self.max_iter = 1_000_000_000
        self.momentum = 0.9
        self.n_features_limit = 2
        self.hidden_layer_sizes = [5, 25, 125]
        self.n_experiments = 2
