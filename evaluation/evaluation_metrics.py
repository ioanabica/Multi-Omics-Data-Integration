import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp
import numpy as np
import pandas as pd
import seaborn as sn
import math

def plot_ROC_curves(ROC_points):

    keys = ROC_points.keys()
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    colors = ['red', 'blue', 'dark orange', 'purple', 'green']
    lw = 2

    for key in keys:
        y_true = ROC_points[key]['y_true']
        y_score = ROC_points[key]['y_score']
        print y_true
        print y_score
        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
        mean_tpr +=  interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=lw, color=colors[key], label='ROC fold %d (area = %0.3f)' % (key, roc_auc))


    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k', label='Random')

    mean_tpr /= len(keys)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='green', linestyle='--', label='Mean ROC (area = %0.3f)' % mean_auc, lw=3)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()


def compute_class_id_to_class_symbol(label_to_one_hot_encoding):
    class_id_to_class_symbol = dict()
    labels = label_to_one_hot_encoding.keys()
    for label in labels:
        class_id = np.argmax(label_to_one_hot_encoding[label])
        class_id_to_class_symbol[class_id] = label
    return class_id_to_class_symbol


def compute_evaluation_metrics_for_each_class(confussion_matrix, class_id_to_class_symbol):

    confussion_matrix = np.array(confussion_matrix)

    class_symbol_to_evaluation_metrics = dict()
    class_ids = class_id_to_class_symbol.keys()

    for class_id in class_ids:
        class_symbol = class_id_to_class_symbol[class_id]
        class_symbol_to_evaluation_metrics[class_symbol] = dict()

    sum_over_rows = confussion_matrix.sum(axis=1)
    sum_over_columns = confussion_matrix.sum(axis=0)


    for class_id in class_ids:
        class_symbol = class_id_to_class_symbol[class_id]

        true_positives = confussion_matrix[class_id][class_id]
        false_positives = sum_over_columns[class_id] - true_positives


        false_negatives = sum_over_rows[class_id] - true_positives
        true_negatives = sum_over_columns.sum() - sum_over_columns[class_id] - \
                         sum_over_rows[class_id] + confussion_matrix[class_id][class_id]


        accuracy = (true_positives + true_negatives)/(true_positives + true_negatives + false_negatives + false_positives)

        if true_positives + false_positives == 0:
            precision = 0
        else:
            precision = true_positives / (true_positives + false_positives)


        if true_positives + false_negatives == 0:
            recall = 0
        else:
            recall = true_positives / (true_positives + false_negatives)

        f1_score = (2 * true_positives) / (2 * true_positives + false_positives + false_negatives)

        MCC = (true_positives * true_negatives - false_positives * false_negatives) / \
              (math.sqrt((true_positives + false_positives) * (true_positives + false_negatives) * \
                         (true_negatives + false_positives) * (true_negatives + false_negatives)))

        class_symbol_to_evaluation_metrics[class_symbol]['true_positives'] = true_positives
        class_symbol_to_evaluation_metrics[class_symbol]['false_positives'] = false_positives

        class_symbol_to_evaluation_metrics[class_symbol]['true_negatives'] = true_negatives
        class_symbol_to_evaluation_metrics[class_symbol]['false_negatives'] = false_negatives

        class_symbol_to_evaluation_metrics[class_symbol]['accuracy'] = precision
        class_symbol_to_evaluation_metrics[class_symbol]['precision'] = precision
        class_symbol_to_evaluation_metrics[class_symbol]['sensitivity'] = recall
        class_symbol_to_evaluation_metrics[class_symbol]['f1_score'] = f1_score
        class_symbol_to_evaluation_metrics[class_symbol]['MCC'] = MCC

    return class_symbol_to_evaluation_metrics


def plot_confussion_matrix_as_heatmap(confussion_matrix, label_to_one_hot_encoding, plt_title):

    confussion_matrix = np.array(confussion_matrix)
    new_confussion_matrix = confussion_matrix

    labels = ['Oocyte', 'Zygote', '2-cell_embryo', '4-cell_embryo', '8-cell_embryo', 'Morulae', 'Late_blastocyst']

    class_id_to_class_label = compute_class_id_to_class_symbol(label_to_one_hot_encoding)
    print label_to_one_hot_encoding
    print class_id_to_class_label

    print new_confussion_matrix

    index = 0
    for label in labels:
        class_id = np.argmax(label_to_one_hot_encoding[label])
        label_to_swap = class_id_to_class_label[index]
        print label
        print class_id
        print label_to_swap
        if(label != label_to_swap):
            class_id_to_swap = index
            print class_id_to_swap
            # swap class_id and class_id_to_swap columns
            for i in range(len(labels)):
                aux = confussion_matrix[i][class_id]
                confussion_matrix[i][class_id] = confussion_matrix[i][class_id_to_swap]
                confussion_matrix[i][class_id_to_swap]=aux

            print confussion_matrix

            # swap class_id and class_id_to_swap rows
            for i in range(len(labels)):
                aux = confussion_matrix[class_id][i]
                confussion_matrix[class_id][i] = confussion_matrix[class_id_to_swap][i]
                confussion_matrix[class_id_to_swap][i] = aux

            print confussion_matrix
            # swap labels
            class_id_to_class_label[class_id] = label_to_swap
            class_id_to_class_label[class_id_to_swap] = label

            #swap one-hot encodings
            aux = label_to_one_hot_encoding[label]
            label_to_one_hot_encoding[label] = label_to_one_hot_encoding[label_to_swap]
            label_to_one_hot_encoding[label_to_swap] = aux

            print class_id_to_class_label
        index +=1


    labels = ['Oocyte', 'Zygote', '2-cell', '4-cell', '8-cell', 'Morulae', 'Late\n blastocyst']
    df_cm = pd.DataFrame(confussion_matrix, index=[i for i in labels], columns=[i for i in labels])

    fig = plt.figure(figsize=(8, 20), dpi=150)
    ax = plt.subplot(111)
    sn.heatmap(df_cm, annot=True)

    ax.xaxis.set_ticks_position('top')
    #plt.xticks(rotation=30)
    plt.yticks(rotation=0)
    plt.title(plt_title, size=18, y=-0.08)
    plt.show()

