import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import stats
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


def plot_mean_ROC_curves(mlp_ROC_points, rnn_ROC_points, snn_ROC_points):

    keys = mlp_ROC_points.keys()
    mlp_mean_tpr = 0.0
    mlp_mean_fpr = np.linspace(0, 1, 100)

    rnn_mean_tpr = 0.0
    rnn_mean_fpr = np.linspace(0, 1, 100)

    snn_mean_tpr = 0.0
    snn_mean_fpr = np.linspace(0, 1, 100)

    colors = ['red', 'blue', 'dark orange', 'purple', 'green']
    lw = 2

    for key in keys:
        # MLP
        y_true = mlp_ROC_points[key]['y_true']
        y_score = mlp_ROC_points[key]['y_score']
        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
        mlp_mean_tpr += interp(mlp_mean_fpr, fpr, tpr)
        mlp_mean_tpr[0] = 0.0

        # RNN
        y_true = rnn_ROC_points[key]['y_true']
        y_score = rnn_ROC_points[key]['y_score']
        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
        rnn_mean_tpr += interp(rnn_mean_fpr, fpr, tpr)
        rnn_mean_tpr[0] = 0.0

        # SNN
        y_true = snn_ROC_points[key]['y_true']
        y_score = snn_ROC_points[key]['y_score']
        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
        snn_mean_tpr += interp(snn_mean_fpr, fpr, tpr)
        snn_mean_tpr[0] = 0.0


    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k', label='Random')

    # MLP
    mlp_mean_tpr /= len(keys)
    mlp_mean_tpr[-1] = 1.0
    mlp_mean_auc = auc(mlp_mean_fpr, mlp_mean_tpr)
    plt.plot(mlp_mean_fpr, mlp_mean_tpr, color='green', linestyle='--', label='Mean ROC for MLP (area = %0.3f)' % mlp_mean_auc, lw=3)

    # RNN
    rnn_mean_tpr /= len(keys)
    rnn_mean_tpr[-1] = 1.0
    rnn_mean_auc = auc(rnn_mean_fpr, rnn_mean_tpr)
    plt.plot(rnn_mean_fpr, rnn_mean_tpr, color='red', linestyle='--', label='Mean ROC for RNN (area = %0.3f)' % rnn_mean_auc,
             lw=3)

    # SNN
    snn_mean_tpr /= len(keys)
    snn_mean_tpr[-1] = 1.0
    snn_mean_auc = auc(snn_mean_fpr, snn_mean_tpr)
    plt.plot(snn_mean_fpr, mlp_mean_tpr, color='blue', linestyle='--', label='Mean ROC for SNN(area = %0.3f)' % snn_mean_auc,
             lw=3)

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

        class_symbol_to_evaluation_metrics[class_symbol]['accuracy'] = accuracy
        class_symbol_to_evaluation_metrics[class_symbol]['precision'] = precision
        class_symbol_to_evaluation_metrics[class_symbol]['sensitivity'] = recall
        class_symbol_to_evaluation_metrics[class_symbol]['f1_score'] = f1_score
        class_symbol_to_evaluation_metrics[class_symbol]['MCC'] = MCC

    print class_symbol_to_evaluation_metrics['cancer']

    return class_symbol_to_evaluation_metrics


def plot_confussion_matrix_as_heatmap(confussion_matrix, label_to_one_hot_encoding, plt_title):

    confussion_matrix = np.array(confussion_matrix)
    new_confussion_matrix = confussion_matrix

    labels = ['Oocyte', 'Zygote', '2-cell_embryo', '4-cell_embryo', '8-cell_embryo', 'Morulae', 'Late_blastocyst']

    class_id_to_class_label = compute_class_id_to_class_symbol(label_to_one_hot_encoding)

    index = 0
    for label in labels:
        class_id = np.argmax(label_to_one_hot_encoding[label])
        label_to_swap = class_id_to_class_label[index]
        if(label != label_to_swap):
            class_id_to_swap = index
            # swap class_id and class_id_to_swap columns
            for i in range(len(labels)):
                aux = confussion_matrix[i][class_id]
                confussion_matrix[i][class_id] = confussion_matrix[i][class_id_to_swap]
                confussion_matrix[i][class_id_to_swap]=aux

            # swap class_id and class_id_to_swap rows
            for i in range(len(labels)):
                aux = confussion_matrix[class_id][i]
                confussion_matrix[class_id][i] = confussion_matrix[class_id_to_swap][i]
                confussion_matrix[class_id_to_swap][i] = aux

            # swap labels
            class_id_to_class_label[class_id] = label_to_swap
            class_id_to_class_label[class_id_to_swap] = label

            #swap one-hot encodings
            aux = label_to_one_hot_encoding[label]
            label_to_one_hot_encoding[label] = label_to_one_hot_encoding[label_to_swap]
            label_to_one_hot_encoding[label_to_swap] = aux

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


def plot_confussion_matrix_as_heatmap_for_cancer_data(confussion_matrix, label_to_one_hot_encoding, plt_title):

    confussion_matrix = np.array(confussion_matrix)

    labels = ['Not Cancer', 'Cancer']

    df_cm = pd.DataFrame(confussion_matrix, index=[i for i in labels], columns=[i for i in labels])

    fig = plt.figure(figsize=(8, 20), dpi=150)
    ax = plt.subplot(111)
    sn.heatmap(df_cm, annot=True)

    ax.xaxis.set_ticks_position('top')
    #plt.xticks(rotation=30)
    plt.yticks(rotation=0)
    plt.title(plt_title, size=18, y=-0.08)
    plt.show()


def compute_average_performance_metrics_for_binary_classification(performance_metrics, cancer_data=True):

    keys = performance_metrics.keys()

    print performance_metrics

    if cancer_data:
        positive_label='cancer'
    else:
        positive_label='positive_class'

    average_performance_metrics = dict()
    std_performance_metrics = dict()

    accuracy = []
    precision = []
    sensitivity = []
    MCC = []
    f1_score = []

    for key in keys:
        fold_performance_metrics = performance_metrics[key]

        accuracy += [fold_performance_metrics[positive_label]['accuracy']]
        precision += [fold_performance_metrics[positive_label]['precision']]
        sensitivity += [fold_performance_metrics[positive_label]['sensitivity']]
        MCC += [fold_performance_metrics[positive_label]['MCC']]
        f1_score += [fold_performance_metrics[positive_label]['f1_score']]

    average_performance_metrics['accuracy'] = np.mean(accuracy)
    average_performance_metrics['precision'] = np.mean(precision)
    average_performance_metrics['sensitivity'] = np.mean(sensitivity)
    average_performance_metrics['MCC'] = np.mean(MCC)
    average_performance_metrics['f1_score'] = np.mean(f1_score)

    std_performance_metrics['accuracy'] = np.std(accuracy)
    std_performance_metrics['precision'] = np.std(precision)
    std_performance_metrics['sensitivity'] = np.std(sensitivity)
    std_performance_metrics['MCC'] = np.std(MCC)
    std_performance_metrics['f1_score'] = np.std(f1_score)

    print average_performance_metrics
    print std_performance_metrics

    return 0


def compute_performance_metrics_for_multiclass_classification(performance_metrics):
    keys = performance_metrics.keys()
    class_symbols = performance_metrics[0].keys()
    print performance_metrics

    class_symbol_to_macro_evaluation_metrics = dict()
    class_symbol_to_macro_evaluation_metrics_std = dict()

    for class_symbol in class_symbols:
        class_symbol_to_macro_evaluation_metrics[class_symbol] = dict()
        class_symbol_to_macro_evaluation_metrics_std[class_symbol] = dict()

    for class_symbol in class_symbols:
        accuracy = []
        precision = []
        sensitivity = []
        MCC = []
        f1_score = []

        for key in keys:
            fold_performance_metrics = performance_metrics[key]

            accuracy += [fold_performance_metrics[class_symbol]['accuracy']]
            precision += [fold_performance_metrics[class_symbol]['precision']]
            sensitivity += [fold_performance_metrics[class_symbol]['sensitivity']]
            MCC += [fold_performance_metrics[class_symbol]['MCC']]
            f1_score += [fold_performance_metrics[class_symbol]['f1_score']]

        class_symbol_to_macro_evaluation_metrics[class_symbol]['accuracy'] = np.mean(accuracy)
        class_symbol_to_macro_evaluation_metrics[class_symbol]['precision'] = np.mean(precision)
        class_symbol_to_macro_evaluation_metrics[class_symbol]['sensitivity'] = np.mean(sensitivity)
        class_symbol_to_macro_evaluation_metrics[class_symbol]['MCC'] = np.mean(MCC)
        class_symbol_to_macro_evaluation_metrics[class_symbol]['f1_score'] = np.mean(f1_score)

        class_symbol_to_macro_evaluation_metrics_std[class_symbol]['accuracy'] = np.std(accuracy)
        class_symbol_to_macro_evaluation_metrics_std[class_symbol]['precision'] = np.std(precision)
        class_symbol_to_macro_evaluation_metrics_std[class_symbol]['sensitivity'] = np.std(sensitivity)
        class_symbol_to_macro_evaluation_metrics_std[class_symbol]['MCC'] = np.std(MCC)
        class_symbol_to_macro_evaluation_metrics_std[class_symbol]['f1_score'] = np.std(f1_score)

    return 0


def paired_t_test_binary_classification(performance_metrics_first_model, performance_metrics_second_model, cancer_data=True):

    keys = performance_metrics_first_model.keys()

    if cancer_data:
        positive_label='cancer'
    else:
        positive_label='positive_class'

    p_values = dict()

    accuracy_first_model = []
    precision_first_model = []
    sensitivity_first_model = []
    MCC_first_model = []
    f1_score_first_model = []

    accuracy_second_model = []
    precision_second_model = []
    sensitivity_second_model = []
    MCC_second_model = []
    f1_score_second_model = []

    for key in keys:
        fold_performance_metrics_first_model = performance_metrics_first_model[key]
        fold_performance_metrics_second_model = performance_metrics_second_model[key]

        accuracy_first_model += [fold_performance_metrics_first_model[positive_label]['accuracy']]
        accuracy_second_model += [fold_performance_metrics_second_model[positive_label]['accuracy']]

        precision_first_model += [fold_performance_metrics_first_model[positive_label]['precision']]
        precision_second_model += [fold_performance_metrics_second_model[positive_label]['precision']]

        sensitivity_first_model += [fold_performance_metrics_first_model[positive_label]['sensitivity']]
        sensitivity_second_model += [fold_performance_metrics_second_model[positive_label]['sensitivity']]

        MCC_first_model += [fold_performance_metrics_first_model[positive_label]['MCC']]
        MCC_second_model += [fold_performance_metrics_second_model[positive_label]['MCC']]

        f1_score_first_model += [fold_performance_metrics_first_model[positive_label]['f1_score']]
        f1_score_second_model += [fold_performance_metrics_second_model[positive_label]['f1_score']]

    print accuracy_first_model
    print accuracy_second_model

    print precision_first_model
    print precision_second_model

    print sensitivity_first_model
    print sensitivity_second_model

    print MCC_first_model
    print MCC_second_model

    print f1_score_first_model
    print f1_score_second_model

    _, p_value = stats.ttest_rel(accuracy_first_model, accuracy_second_model)
    p_values['accuracy'] = p_value

    _, p_value = stats.ttest_rel(precision_first_model, precision_second_model)
    p_values['precision'] = p_value

    _, p_value = stats.ttest_rel(sensitivity_first_model, sensitivity_second_model)
    p_values['sensitivity'] = p_value

    _, p_value = stats.ttest_rel(f1_score_first_model, f1_score_second_model)
    p_values['f1_score'] = p_value

    _, p_value = stats.ttest_rel(MCC_first_model, MCC_second_model)
    p_values['MCC'] = p_value

    print p_values

    return p_values