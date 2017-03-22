from embryo_development_data_processing import extract_data_from_embryo_stage_file, extract_embryo_id_to_gene_expressions, \
    extract_embryo_id_to_gene_expressions_clusters, extract_gene_id_to_gene_entropy_and_expression_levels, \
    compute_clusters_size, create_one_hot_encoding, create_embryo_stage_to_embryo_ids
from embryo_development_datasets import create_k_fold_embryo_ids, \
    create_k_fold_datasets, create_k_fold_datasets_with_clusters
from epigenetic_data.epigenetic_data import EpigeneticData
from gene_clustering.hierarchical_clustering import hierarchical_clustering
from gene_clustering.k_means_clusters import k_means_clustering

"""
Class that extracts the data to perform supervised learning using the neural network architectures.
"""


class EmbryoDevelopmentData(EpigeneticData):

    def __init__(self, num_folds, num_folds_hyperparameters_tuning, max_num_genes, gene_entropy_threshold):
        EpigeneticData.__init__(self, num_folds, num_folds_hyperparameters_tuning)
        self.max_num_genes = max_num_genes
        self.gene_entropy_threshold = gene_entropy_threshold
        self.extract_data_from_gene_expression_file()
        self.extract_data_from_embryo_stage_file()
        self.input_data_size = max_num_genes
        self.output_size = len(self.embryo_stage_to_embryo_ids.keys())

    def get_k_fold_datasets(self):
        embryoIds = self.embryo_id_to_embryo_stage.keys()

        embryo_stages = self.embryo_stage_to_embryo_ids.keys()
        self.embryo_stage_to_one_hot_encoding = create_one_hot_encoding(embryo_stages)

        k_fold_embryo_ids = create_k_fold_embryo_ids(self.num_folds, self.embryo_stage_to_embryo_ids)

        print k_fold_embryo_ids
        self.k_fold_datasets = create_k_fold_datasets(
            self.num_folds, k_fold_embryo_ids, self.input_data_size, self.output_size,
            self.embryo_id_to_gene_expressions, self.embryo_stage_to_one_hot_encoding, self.embryo_id_to_embryo_stage)

        self.k_fold_datasets_hyperparameters_tuning = dict()

        for index_i in range(self.num_folds):
            training_embryo_ids = []
            for index_j in range(self.num_folds):
                if index_j != index_i:
                    training_embryo_ids += k_fold_embryo_ids[index_j]

            embryo_stage_to_embryo_ids = create_embryo_stage_to_embryo_ids(
                training_embryo_ids, self.embryo_id_to_embryo_stage)
            k_fold_embryo_ids_hyperparameters_tuning = create_k_fold_embryo_ids(
                self.num_folds_hyperparameters_tuning, embryo_stage_to_embryo_ids)

            print k_fold_embryo_ids_hyperparameters_tuning

            k_fold_dataset = create_k_fold_datasets(
                self.num_folds_hyperparameters_tuning, k_fold_embryo_ids_hyperparameters_tuning,
                self.input_data_size, self.output_size,
                self.embryo_id_to_gene_expressions, self.embryo_stage_to_one_hot_encoding,
                self.embryo_id_to_embryo_stage)

            self.k_fold_datasets_hyperparameters_tuning[index_i] = k_fold_dataset

        return self.k_fold_datasets, self.k_fold_datasets_hyperparameters_tuning

    def extract_data_from_embryo_stage_file(self):
        embryo_stage_file = open(
            '/home/ioana/PycharmProjects/Part-II-Project/datasets/human_early_embryo_stage.txt', 'r')
        self.embryo_id_to_embryo_stage, self.embryo_stage_to_embryo_ids = \
            extract_data_from_embryo_stage_file(embryo_stage_file)
        embryo_stage_file.close()

    def extract_data_from_gene_expression_file(self):
        embryo_gene_expressions_file = open(
            '/home/ioana/PycharmProjects/Part-II-Project/datasets/human_early_embryo_gene_expression.txt', 'r')
        self.geneId_to_gene_entropy, self.geneId_to_expression_levels = \
            extract_gene_id_to_gene_entropy_and_expression_levels(
                embryo_gene_expressions_file, self.gene_entropy_threshold, self.max_num_genes)

        embryo_gene_expressions_file.seek(0)
        self.embryo_id_to_gene_expressions = extract_embryo_id_to_gene_expressions(
            embryo_gene_expressions_file, self.geneId_to_gene_entropy, self.gene_entropy_threshold, self.max_num_genes)

        embryo_gene_expressions_file.close()


class EmbryoDevelopmentDataWithClusters(EmbryoDevelopmentData):

    def __init__(self,
                 num_clusters, clustering_algorithm,
                 num_folds, num_folds_hyperparameters_tuning,
                 max_num_genes, gene_entropy_threshold):
        EmbryoDevelopmentData.__init__(
            self, num_folds, num_folds_hyperparameters_tuning, max_num_genes, gene_entropy_threshold)

        self.num_clusters = num_clusters
        self.extract_clustering_data_from_gene_expression_file(clustering_algorithm)
        self.output_size = len(self.embryo_stage_to_embryo_ids.keys())

    def get_k_fold_datasets(self):

        embryo_stages = self.embryo_stage_to_embryo_ids.keys()
        self.embryo_stage_to_one_hot_encoding = create_one_hot_encoding(embryo_stages)

        k_fold_embryo_ids = create_k_fold_embryo_ids(self.num_folds, self.embryo_stage_to_embryo_ids)
        print k_fold_embryo_ids

        self.k_fold_datasets = create_k_fold_datasets_with_clusters(
            self.num_folds, k_fold_embryo_ids,
            self.clusters_size, self.output_size,
            self.embryo_id_to_gene_expressions_clusters,
            self.embryo_stage_to_one_hot_encoding, self.embryo_id_to_embryo_stage)

        self.k_fold_datasets_hyperparameters_tuning = dict()

        for index_i in range(self.num_folds):
            training_embryo_ids = []
            for index_j in range(self.num_folds):
                if index_j != index_i:
                    training_embryo_ids += k_fold_embryo_ids[index_j]

            embryo_stage_to_embryo_ids  = create_embryo_stage_to_embryo_ids(
                training_embryo_ids, self.embryo_id_to_embryo_stage)
            k_fold_embryo_ids_hyperparameters_tuning = create_k_fold_embryo_ids(
                self.num_folds_hyperparameters_tuning, embryo_stage_to_embryo_ids)

            print k_fold_embryo_ids_hyperparameters_tuning

            k_fold_dataset = create_k_fold_datasets_with_clusters(
                self.num_folds_hyperparameters_tuning, k_fold_embryo_ids_hyperparameters_tuning,
                self.clusters_size, self.output_size,
                self.embryo_id_to_gene_expressions_clusters,
                self.embryo_stage_to_one_hot_encoding, self.embryo_id_to_embryo_stage)

            self.k_fold_datasets_hyperparameters_tuning[index_i] = k_fold_dataset

        return self.k_fold_datasets, self.k_fold_datasets_hyperparameters_tuning

    def extract_clustering_data_from_gene_expression_file(self, clustering_algorithm):
        embryo_gene_expressions_file = open(
            '/home/ioana/PycharmProjects/Part-II-Project/datasets/human_early_embryo_gene_expression.txt', 'r')

        if clustering_algorithm == 'hierarchical':
            self.gene_id_to_gene_cluster, self.gene_clusters = \
                hierarchical_clustering(self.geneId_to_expression_levels, self.num_clusters)
        else:
            self.gene_id_to_gene_cluster, self.gene_clusters = \
                k_means_clustering(self.geneId_to_expression_levels, self.num_clusters)


        self.embryo_id_to_gene_expressions_clusters = extract_embryo_id_to_gene_expressions_clusters(
            embryo_gene_expressions_file, self.gene_id_to_gene_cluster)

        embryo_gene_expressions_file.close()

        self.clusters_size = compute_clusters_size(self.gene_clusters)



class EmbryoDevelopmentDataWithSingleCluster(EmbryoDevelopmentDataWithClusters):

    def __init__(self,
                 num_clusters, clustering_algorithm,
                 num_folds, num_folds_hyperparameters_tuning,
                 max_num_genes, gene_entropy_threshold):
        EmbryoDevelopmentDataWithClusters.__init__(
            self, num_clusters, clustering_algorithm,
                 num_folds, num_folds_hyperparameters_tuning,
                 max_num_genes, gene_entropy_threshold)

    def get_k_fold_datasets(self):

        self.input_data_size = self.clusters_size[0]

        embryo_id_to_gene_expressions = dict()

        for embryo_id in self.embryo_id_to_gene_expressions_clusters.keys():
            embryo_id_to_gene_expressions[embryo_id] = self.embryo_id_to_gene_expressions_clusters[embryo_id][0]
            print "got here"
            embryo_id_to_gene_expressions[embryo_id] = embryo_id_to_gene_expressions[embryo_id][0:128]

        print self.clusters_size

        embryo_stages = self.embryo_stage_to_embryo_ids.keys()
        self.embryo_stage_to_one_hot_encoding = create_one_hot_encoding(embryo_stages)

        k_fold_embryo_ids = create_k_fold_embryo_ids(self.num_folds, self.embryo_stage_to_embryo_ids)

        print k_fold_embryo_ids
        self.k_fold_datasets = create_k_fold_datasets(
            self.num_folds, k_fold_embryo_ids, self.input_data_size, self.output_size,
            embryo_id_to_gene_expressions, self.embryo_stage_to_one_hot_encoding, self.embryo_id_to_embryo_stage)

        self.k_fold_datasets_hyperparameters_tuning = dict()

        for index_i in range(self.num_folds):
            training_embryo_ids = []
            for index_j in range(self.num_folds):
                if index_j != index_i:
                    training_embryo_ids += k_fold_embryo_ids[index_j]

            embryo_stage_to_embryo_ids = create_embryo_stage_to_embryo_ids(
                training_embryo_ids, self.embryo_id_to_embryo_stage)
            k_fold_embryo_ids_hyperparameters_tuning = create_k_fold_embryo_ids(
                self.num_folds_hyperparameters_tuning, embryo_stage_to_embryo_ids)

            print k_fold_embryo_ids_hyperparameters_tuning

            k_fold_dataset = create_k_fold_datasets(
                self.num_folds_hyperparameters_tuning, k_fold_embryo_ids_hyperparameters_tuning,
                self.input_data_size, self.output_size,
                embryo_id_to_gene_expressions, self.embryo_stage_to_one_hot_encoding,
                self.embryo_id_to_embryo_stage)

            self.k_fold_datasets_hyperparameters_tuning[index_i] = k_fold_dataset

        return self.k_fold_datasets, self.k_fold_datasets_hyperparameters_tuning
