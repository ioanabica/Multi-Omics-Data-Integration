
class EpigeneticData:

    def __init__(self, num_folds, num_folds_hyperparameters_tuning):
        self.num_folds = num_folds
        self.num_folds_hyperparameters_tuning = num_folds_hyperparameters_tuning
        self.k_fold_datasets = dict()
        self.k_fold_datasets_hyperparameters_tuning = dict()



