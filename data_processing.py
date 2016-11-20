import os
import numpy
import tensorflow
import scipy.stats
import math

# Set to true for a verbose output
DEBUG = False


def compute_probability_distribution(gene_expressions):
    gene_expressions_sum = 0.0

    for gene_expression in gene_expressions:
        gene_expressions_sum += float(gene_expression)

    for gene_expression in gene_expressions:
        gene_expression = float(gene_expression)/gene_expressions_sum

    return gene_expressions

def compute_gene_entropy(gene_expressions):
    gene_entropy = 0.0

    for gene_expression in gene_expressions:
        if (float(gene_expression) > 0.0):
            gene_entropy -= float(gene_expression) * math.log(float(gene_expression), 2)

    return gene_entropy

"""
Class that processes the epigenetics dataset
"""
class EpigeneticsDataExtractor(object):

    gene_expressions_file = open('data/epigenetics_data/human_early_embryo_gene_expression.txt', 'r')
    embryo_stage_file = open('data/epigenetics_data/human_early_embryo_stage.txt', 'r')
    #dataset =  numpy.ndarray(shape=(number_of_classes, ))

    def extract_embryoId_to_embryoStage(file):
        embryoId_to_embryoStage = dict()
        file.readline()
        for line in file:
            line_elements = line.split()
            embryoId_to_embryoStage[line_elements[0]] = line_elements[1]
        return embryoId_to_embryoStage

    embryoId_to_embryoStage = extract_embryoId_to_embryoStage(embryo_stage_file)
    print(embryoId_to_embryoStage)

    def extract_gene_expressions (file):
        geneId_to_geneExpressions = dict()

        # extract the embryo_Ids from the first line in the file
        embryo_Ids = file.readline().split()
        embryo_Ids = embryo_Ids[1:]

        for line in file:
            line_elements = line.split()
            geneId_to_geneExpressions[line_elements[0]] = line_elements[1:]
            gene_entropy = compute_gene_entropy(compute_probability_distribution(line_elements[1:]))
            print gene_entropy
        return geneId_to_geneExpressions

    geneId_to_geneExpressions = extract_gene_expressions(gene_expressions_file)
    print(compute_gene_entropy([0.5, 0.5]))

 #   train_data + train_labels
 #   validation_data + validation_labels
 #   test_data + test_labels


