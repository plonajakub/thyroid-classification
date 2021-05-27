class Parameters:
    def __init__(self):
        # Tested parameters
        self.resample = True
        self.n_features = 5
        self.hidden_layer_sizes = (100,)
        self.learning_rate_init = 0.1
        self.learning_rate = 'adaptive'
        self.momentum = 0.9

        # Tests accuracy
        self.max_iter = 10000
        self.n_experiments = 5

        # Misc params
        self.scoring = 'balanced_accuracy'
        self.random_state = 0
        self.n_jobs = -1
        self.verbose = 0

        # Constants
        self.nesterovs_momentum = False
        self.cv = 2
        self.solver = 'sgd'
