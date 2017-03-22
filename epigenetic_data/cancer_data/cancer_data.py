from cancer_data_datasets import create_k_fold_patient_ids, \
    create_k_fold_datasets, create_k_fold_datasets_with_clusters

from cancer_data_processing import extract_patients_data, create_one_hot_encoding, create_label_to_patient_ids
from epigenetic_data.epigenetic_data import EpigeneticData


class CancerData(EpigeneticData):

    def __init__(self, num_folds, num_folds_hyperparameters_tuning):
        EpigeneticData.__init__(self, num_folds, num_folds_hyperparameters_tuning)
        self.extract_data_from_cancer_data_file()

    def extract_data_from_cancer_data_file(self):
        cancer_data_file = open('data/epigenetics_data/expr_methyl_data_TNF.out', 'r')
        self.patient_id_to_gene_expressions, self.patient_id_to_dna_methilation, \
            self.patient_id_to_gene_expressions_and_dna_methylation, self.patient_id_to_label, self.label_to_patient_ids = \
            extract_patients_data(cancer_data_file)

    def get_k_fold_datasets(self):

        patient_ids = self.patient_id_to_label.keys()

        self.input_data_size = len(self.patient_id_to_gene_expressions_and_dna_methylation[patient_ids[0]])

        labels = self.label_to_patient_ids.keys()
        self.label_to_one_hot_encoding = create_one_hot_encoding(labels)
        self.output_size = len(labels)

        k_fold_patient_ids = create_k_fold_patient_ids(self.num_folds, self.label_to_patient_ids)
        print k_fold_patient_ids

        self.k_fold_datasets = create_k_fold_datasets(
            self.num_folds, k_fold_patient_ids, self.input_data_size, self.output_size,
            self.patient_id_to_gene_expressions_and_dna_methylation, self.label_to_one_hot_encoding,
            self.patient_id_to_label)

        self.k_fold_datasets_hyperparameters_tuning = dict()

        for index_i in range(self.num_folds):
            training_patient_ids = []
            for index_j in range(self.num_folds):
                if index_j != index_i:
                    training_patient_ids += k_fold_patient_ids[index_j]

            labels_to_patient_ids = create_label_to_patient_ids(
                training_patient_ids, self.patient_id_to_label)

            k_fold_patient_ids_hyperparameters_tuning = create_k_fold_patient_ids(
                self.num_folds_hyperparameters_tuning, labels_to_patient_ids)

            k_fold_dataset = create_k_fold_datasets(
                self.num_folds_hyperparameters_tuning, k_fold_patient_ids_hyperparameters_tuning,
                self.input_data_size, self.output_size,
                self.patient_id_to_gene_expressions_and_dna_methylation, self.label_to_one_hot_encoding,
                self.patient_id_to_label)

            self.k_fold_datasets_hyperparameters_tuning[index_i] = k_fold_dataset

        return self.k_fold_datasets, self.k_fold_datasets_hyperparameters_tuning


#data = CancerData(3)

#k_fold_datasets = data.get_k_fold_datasets()
#training_dataset, validation_dataset, test_dataset = data.get_training_validation_test_datasets()


class CancerDataWithClusters(CancerData):

    def __init__(self, num_folds, num_folds_hyperparameters_tuning):
        CancerData.__init__(self, num_folds, num_folds_hyperparameters_tuning)
        self.num_clusters = 2
        self.patient_id_to_input_clusters = self.create_patient_id_to_input_clusters()

        """
            The cancer data contains the gene_expressions and the dna_methylation for each patient, which we will
            the clusters.
        """

    def create_patient_id_to_input_clusters(self):

        patient_id_to_input_clusters = dict()

        patient_ids = self.patient_id_to_dna_methilation.keys()

        for patient_id in patient_ids:
            patient_id_to_input_clusters[patient_id] = dict()

        for patient_id in patient_ids:
            patient_id_to_input_clusters[patient_id][0] = self.patient_id_to_gene_expressions[patient_id]
            patient_id_to_input_clusters[patient_id][1] = self.patient_id_to_dna_methilation[patient_id]

        return patient_id_to_input_clusters

    def get_k_fold_datasets(self):

        patient_ids = self.patient_id_to_label.keys()

        self.clusters_size = [len(self.patient_id_to_gene_expressions[patient_ids[0]]),
                              len(self.patient_id_to_dna_methilation[patient_ids[0]])]

        labels = self.label_to_patient_ids.keys()
        self.label_to_one_hot_encoding = create_one_hot_encoding(labels)
        self.output_size = len(labels)

        self.k_fold_patient_ids = create_k_fold_patient_ids(self.num_folds, self.label_to_patient_ids)

        self.k_fold_datasets = create_k_fold_datasets_with_clusters(
            self.num_folds, self.k_fold_patient_ids, self.clusters_size, self.output_size,
            self.patient_id_to_input_clusters, self.label_to_one_hot_encoding,
            self.patient_id_to_label)

        self.k_fold_datasets_hyperparameters_tuning = dict()

        for index_i in range(self.num_folds):
            training_patient_ids = []
            for index_j in range(self.num_folds):
                if index_j != index_i:
                    training_patient_ids += self.k_fold_patient_ids[index_j]

            labels_to_patient_ids = create_label_to_patient_ids(
                training_patient_ids, self.patient_id_to_label)

            k_fold_patient_ids_hyperparameters_tuning = create_k_fold_patient_ids(
                self.num_folds_hyperparameters_tuning, labels_to_patient_ids)

            k_fold_dataset = create_k_fold_datasets_with_clusters(
                self.num_folds, k_fold_patient_ids_hyperparameters_tuning, self.clusters_size, self.output_size,
                self.patient_id_to_input_clusters, self.label_to_one_hot_encoding,
                self.patient_id_to_label)

            self.k_fold_datasets_hyperparameters_tuning[index_i] = k_fold_dataset

        return self.k_fold_datasets, self.k_fold_datasets_hyperparameters_tuning
