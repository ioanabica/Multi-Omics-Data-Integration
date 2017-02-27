
class EpigeneticData:

    def __init__(self, num_folds):
        self.num_folds = num_folds
        self.k_fold_datasets = dict()
        self.training_dataset = dict()
        self.validation_dataset = dict()
        self.test_dataset = dict()



