import math

# Set to true for a verbose output
DEBUG = False

# normalize the gene expressions to obtain a probability distribution
def compute_probability_distribution(gene_expressions):
    gene_expressions_sum = 0.0
    for gene_expression in gene_expressions:
        gene_expressions_sum += float(gene_expression)
    normalized_gene_expressions = range(len(gene_expressions))
    for index in range(len(gene_expressions)):
        normalized_gene_expressions[index] = float(gene_expressions[index])/gene_expressions_sum

    return normalized_gene_expressions

def compute_gene_entropy(gene_expressions):
    gene_entropy = 0.0
    for gene_expression in gene_expressions:
        if float(gene_expression) > 0.0:
            gene_entropy -= float(gene_expression) * math.log(float(gene_expression), 2)

    return gene_entropy

# create a dictionary from an embryoId to the corresponding development stage
def extract_embryoId_to_embryoStage(file):
    embryoId_to_embryoStage = dict()
    file.readline()
    for line in file:
        line_elements = line.split()
        embryoId_to_embryoStage[line_elements[0]] = line_elements[1]
    return embryoId_to_embryoStage

# create a dictionary from an embryo development stage to a list of the corresponding embroyIds
def extract_embryoStage_to_embryoIds(file):
    embryoStage_to_embryoIds = dict()
    file.readline()
    for line in file:
        line_elements = line.split()
        if line_elements[1] in embryoStage_to_embryoIds.keys():
            embryoStage_to_embryoIds[line_elements[1]] += [line_elements[0]]
        else:
            embryoStage_to_embryoIds[line_elements[1]] = [line_elements[0]]
    return embryoStage_to_embryoIds

def extract_geneId_to_geneEntorpy(file):
    geneId_to_geneEntropy = dict()
    file.readline()
    for line in file:
        line_elements = line.split()
        gene_entropy = compute_gene_entropy(compute_probability_distribution(line_elements[1:]))
        geneId_to_geneEntropy[line_elements[0]] = gene_entropy

    return geneId_to_geneEntropy

def extract_embryoId_to_geneExpressions (file, geneId_to_geneEntropy):
    embryoId_to_geneExpressions = dict()

    #read first line and create an entry in the dictionary for each embryoId
    embryoIds = (file.readline()).split()
    embryoIds = embryoIds[1:]

    for embryoId in embryoIds:
        embryoId_to_geneExpressions[embryoId] = []

    for line in file:
        line_elements = line.split()
        if (geneId_to_geneEntropy[line_elements[0]] > 6.0) & (len(line_elements) == len(embryoIds) + 1):
            for index in range(len(embryoIds)):
                embryoId_to_geneExpressions[embryoIds[index]] += [line_elements[index+1]]

    return embryoId_to_geneExpressions

def create_oneHotEncoding(embryoStages):
    embryoStage_to_oneHotEncoding = dict()

    for index in range(len(embryoStages)):
        oneHotEncoding = [0.0]*len(embryoStages)
        oneHotEncoding[index] = 1.0
        embryoStage_to_oneHotEncoding[embryoStages[index]] = oneHotEncoding

    return embryoStage_to_oneHotEncoding

"""
Class that processes the epigenetics dataset
"""
class EpigeneticsData(object):

    gene_expressions_file = open('data/epigenetics_data/human_early_embryo_gene_expression.txt', 'r')
    embryo_stage_file = open('data/epigenetics_data/human_early_embryo_stage.txt', 'r')

    embryoId_to_embryoStage = extract_embryoId_to_embryoStage(embryo_stage_file)
    embryo_stage_file.seek(0)
    embryoStage_to_embryoIds = extract_embryoStage_to_embryoIds(embryo_stage_file)

    geneId_to_geneEntropy = extract_geneId_to_geneEntorpy(gene_expressions_file)
    gene_expressions_file.seek(0)
    embryoId_to_geneExpressions = extract_embryoId_to_geneExpressions(gene_expressions_file, geneId_to_geneEntropy)

    gene_expressions_file.close()
    embryo_stage_file.close()

    embryoStages = embryoStage_to_embryoIds.keys()
    embryoStage_to_oneHotEncoding = create_oneHotEncoding(embryoStages)
    print embryoStage_to_oneHotEncoding

    training_embryoIds = []
    training_data = []
    training_labels = []

    validation_embryoIds = []
    validation_data = []
    validation_labels = []

    test_embryoIds = []
    test_data = []
    test_labels = []

    embryoStages = embryoStage_to_embryoIds.keys()
    for embryoStage in embryoStages:
        embryoIds = embryoStage_to_embryoIds[embryoStage]
        if len(embryoIds) < 6:
            test_embryoIds += [embryoIds[0]]
            validation_embryoIds += [embryoIds[1]]
            training_embryoIds += embryoIds[2:]
        else:
            test_embryoIds += embryoIds[0:2]
            validation_embryoIds += embryoIds[2:4]
            training_embryoIds += embryoIds[4:]

    for embryoId in training_embryoIds:
        training_data += embryoId_to_geneExpressions[embryoId]
        training_labels += [embryoStage_to_oneHotEncoding[embryoId_to_embryoStage[embryoId]]]

    for embryoId in validation_embryoIds:
        validation_data += embryoId_to_geneExpressions[embryoId]
        validation_labels += [embryoStage_to_oneHotEncoding[embryoId_to_embryoStage[embryoId]]]

    for embryoId in training_embryoIds:
        test_data += embryoId_to_geneExpressions[embryoId]
        test_labels += [embryoStage_to_oneHotEncoding[embryoId_to_embryoStage[embryoId]]]

