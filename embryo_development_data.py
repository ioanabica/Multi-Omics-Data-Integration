from embryo_development_data_processing import *
from epigenetic_data import EpigeneticData
from gene_clustering import plot_dendogram, k_means_clustering


"""
Class that extracts the data to perform supervised learning using the neural network architectures.
"""


class EmbryoDevelopmentData(EpigeneticData):

    def __init__(self, num_folds, max_num_genes, gene_entropy_threshold):
        EpigeneticData.__init__(self, num_folds)
        self.max_num_genes = max_num_genes
        self.gene_entropy_threshold = gene_entropy_threshold
        self.extract_data_from_gene_expression_file()
        self.extract_data_from_embryo_stage_file()

    def get_k_fold_datasets(self):
        embryoIds = self.embryo_id_to_embryo_stage.keys()

        self.input_data_size = len(self.embryo_id_to_gene_expressions[embryoIds[0]])

        embryo_stages = self.embryo_stage_to_embryo_ids.keys()
        self.embryo_stage_to_one_hot_encoding = create_one_hot_encoding(embryo_stages)

        self.output_size = len(embryo_stages)

        self.k_fold_embryoIds = create_k_fold_embryo_ids(self.num_folds, self.embryo_stage_to_embryo_ids)
        print "non-clusters"
        print self.k_fold_embryoIds

        self.k_fold_datasets = create_k_fold_datasets(
            self.num_folds, self.k_fold_embryoIds, self.input_data_size, self.output_size,
            self.embryo_id_to_gene_expressions, self.embryo_stage_to_one_hot_encoding, self.embryo_id_to_embryo_stage)

        return self.k_fold_datasets

    def get_training_validation_test_datasets(self):
        training_embryo_ids, validation_embryo_ids, test_embryo_ids = \
            extract_training_validation_test_embryo_ids(self.embryo_stage_to_embryo_ids)

        embryoIds = self.embryo_id_to_embryo_stage.keys()
        self.input_data_size = len(self.embryo_id_to_gene_expressions[embryoIds[0]])

        embryo_stages = self.embryo_stage_to_embryo_ids.keys()
        self.embryo_stage_to_one_hot_encoding = create_one_hot_encoding(embryo_stages)

        self.output_size = len(embryo_stages)

        training_dataset = create_training_dataset(
            training_embryo_ids,
            self.input_data_size,
            self.output_size,
            self.embryo_id_to_gene_expressions,
            self.embryo_stage_to_one_hot_encoding,
            self.embryo_id_to_embryo_stage)

        validation_dataset = create_validation_dataset(
            validation_embryo_ids,
            self.input_data_size,
            self.output_size,
            self.embryo_id_to_gene_expressions,
            self.embryo_stage_to_one_hot_encoding,
            self.embryo_id_to_embryo_stage)

        test_dataset = create_test_dataset(
            test_embryo_ids,
            self.input_data_size,
            self.output_size,
            self.embryo_id_to_gene_expressions,
            self.embryo_stage_to_one_hot_encoding,
            self.embryo_id_to_embryo_stage)

        return training_dataset, validation_dataset, test_dataset

    def extract_data_from_embryo_stage_file(self):
        embryo_stage_file = open('data/epigenetics_data/human_early_embryo_stage.txt', 'r')
        self.embryo_id_to_embryo_stage, self.embryo_stage_to_embryo_ids = \
            extract_data_from_embryo_stage_file(embryo_stage_file)
        embryo_stage_file.close()

    def extract_data_from_gene_expression_file(self):
        embryo_gene_expressions_file = open('data/epigenetics_data/human_early_embryo_gene_expression.txt', 'r')
        self.geneId_to_gene_entropy, self.geneId_to_expression_levels = \
            extract_gene_id_to_gene_entropy_and_expression_levels(
                embryo_gene_expressions_file, self.gene_entropy_threshold, self.max_num_genes)

        embryo_gene_expressions_file.seek(0)
        self.embryo_id_to_gene_expressions = extract_embryo_id_to_gene_expressions(
            embryo_gene_expressions_file, self.geneId_to_gene_entropy, self.gene_entropy_threshold, self.max_num_genes)

        embryo_gene_expressions_file.close()


class EmbryoDevelopmentDataWithClusters(EmbryoDevelopmentData):

    def __init__(self, num_clusters, num_folds, max_num_genes, gene_entropy_threshold):
        EmbryoDevelopmentData.__init__(self, num_folds, max_num_genes, gene_entropy_threshold)
        self.num_clusters = num_clusters
        self.extract_clustering_data_from_gene_expression_file()

    def extract_clustering_data_from_gene_expression_file(self):
        embryo_gene_expressions_file = open('data/epigenetics_data/human_early_embryo_gene_expression.txt', 'r')

        self.gene_id_to_gene_cluster, self.gene_clusters = \
            hierarchical_clustering(self.geneId_to_expression_levels, self.num_clusters)


        self.embryo_id_to_gene_expressions_clusters = extract_embryo_id_to_gene_expressions_clusters(
            embryo_gene_expressions_file, self.gene_id_to_gene_cluster)

        embryo_gene_expressions_file.close()

        self.clusters_size = compute_clusters_size(self.gene_clusters)

    def get_k_fold_datasets(self):
        embryoIds = self.embryo_id_to_embryo_stage.keys()

        self.input_data_size = len(self.embryo_id_to_gene_expressions[embryoIds[0]])

        embryo_stages = self.embryo_stage_to_embryo_ids.keys()
        self.embryo_stage_to_one_hot_encoding = create_one_hot_encoding(embryo_stages)

        self.output_size = len(embryo_stages)

        self.k_fold_embryoIds = create_k_fold_embryo_ids(self.num_folds, self.embryo_stage_to_embryo_ids)
        print self.k_fold_embryoIds

        self.k_fold_datasets = create_k_fold_datasets_with_clusters(
            self.num_folds, self.k_fold_embryoIds, self.clusters_size, self.output_size,
            self.embryo_id_to_gene_expressions_clusters, self.embryo_stage_to_one_hot_encoding, self.embryo_id_to_embryo_stage)


        return self.k_fold_datasets


    def get_training_validation_test_datasets(self):
        training_embryo_ids, validation_embryo_ids, test_embryo_ids = \
            extract_training_validation_test_embryo_ids(self.embryo_stage_to_embryo_ids)

        embryoIds = self.embryo_id_to_embryo_stage.keys()
        self.input_data_size = len(self.embryo_id_to_gene_expressions[embryoIds[0]])

        embryo_stages = self.embryo_stage_to_embryo_ids.keys()
        self.embryo_stage_to_one_hot_encoding = create_one_hot_encoding(embryo_stages)

        self.output_size = len(embryo_stages)

        training_dataset = create_training_dataset_with_clusters(
            training_embryo_ids,
            self.clusters_size,
            self.output_size,
            self.embryo_id_to_gene_expressions_clusters,
            self.embryo_stage_to_one_hot_encoding,
            self.embryo_id_to_embryo_stage)

        validation_dataset = create_validation_dataset_with_clusters(
            validation_embryo_ids,
            self.clusters_size,
            self.output_size,
            self.embryo_id_to_gene_expressions_clusters,
            self.embryo_stage_to_one_hot_encoding,
            self.embryo_id_to_embryo_stage)

        test_dataset = create_test_dataset_with_clusters(
            test_embryo_ids,
            self.clusters_size,
            self.output_size,
            self.embryo_id_to_gene_expressions_clusters,
            self.embryo_stage_to_one_hot_encoding,
            self.embryo_id_to_embryo_stage)

        return training_dataset, validation_dataset, test_dataset