import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import stats
from scipy import interp
import numpy as np

import math

def plot_ROC_curves(ROC_points):

    keys = ROC_points.keys()
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    fig = plt.figure(figsize=(11, 7), dpi=150)
    ax = plt.subplot(111)
    colors = ['r', 'g', 'b', 'm', 'y', 'c', 'pink', 'orange', 'indigo', 'k']
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
    #plt.plot(mean_fpr, mean_tpr, color='green', linestyle='--', label='Mean ROC (area = %0.3f)' % mean_auc, lw=3)

    plt.xlim([0.01, 1.01])
    plt.ylim([0.01, 1.01])

    plt.xlabel('False Positive Rate', size='24')
    plt.ylabel('True Positive Rate', size='24')

    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.65, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.5, 0.9), shadow=False, ncol=1, prop={'size': 10})

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
    lw = 3

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


    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k', label='Random (AUC = 0.500)')

    # MLP
    mlp_mean_tpr /= len(keys)
    mlp_mean_tpr[-1] = 1.0
    mlp_mean_auc = auc(mlp_mean_fpr, mlp_mean_tpr)
    plt.plot(mlp_mean_fpr, mlp_mean_tpr, color='green', label='MLP (AUC = %0.3f)' % mlp_mean_auc, lw=3)

    # RNN
    rnn_mean_tpr /= len(keys)
    rnn_mean_tpr[-1] = 1.0
    rnn_mean_auc = auc(rnn_mean_fpr, rnn_mean_tpr)
    plt.plot(rnn_mean_fpr, rnn_mean_tpr, color='red', label='RNN (AUC = %0.3f)' % rnn_mean_auc,
             lw=3)

    # SNN
    snn_mean_tpr /= len(keys)
    snn_mean_tpr[-1] = 1.0
    snn_mean_auc = auc(snn_mean_fpr, snn_mean_tpr)
    plt.plot(snn_mean_fpr, snn_mean_tpr, color='blue', label='SNN (AUC = %0.3f)' % snn_mean_auc,
             lw=3)

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])

    plt.xlabel('False Positive Rate', size=24)
    plt.ylabel('True Positive Rate', size=24)
    plt.legend(loc='lower right', prop={'size': 24})

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)


    plt.title('Mean ROC curves after 10-fold outer CV', size=28)

    plt.show()


def plot_mean_ROC_curves_for_two_models(mlp_ROC_points, rnn_ROC_points):

    keys = mlp_ROC_points.keys()
    mlp_mean_tpr = 0.0
    mlp_mean_fpr = np.linspace(0, 1, 100)

    rnn_mean_tpr = 0.0
    rnn_mean_fpr = np.linspace(0, 1, 100)

    colors = ['red', 'blue', 'dark orange', 'purple', 'green']
    lw = 3

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


    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k', label='Random (AUC = 0.500)')

    # MLP
    mlp_mean_tpr /= len(keys)
    mlp_mean_tpr[-1] = 1.0
    mlp_mean_auc = auc(mlp_mean_fpr, mlp_mean_tpr)
    plt.plot(mlp_mean_fpr, mlp_mean_tpr, color='green', label='RNN trained on DNA methylation\nand gene expression data (AUC = %0.3f)' % mlp_mean_auc, lw=3)

    # RNN
    rnn_mean_tpr /= len(keys)
    rnn_mean_tpr[-1] = 1.0
    rnn_mean_auc = auc(rnn_mean_fpr, rnn_mean_tpr)
    plt.plot(rnn_mean_fpr, rnn_mean_tpr, color='red', label='RNN trained on DNA methylation data (AUC = %0.3f)' % rnn_mean_auc,
             lw=3)

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])

    plt.xlabel('False Positive Rate', size=24)
    plt.ylabel('True Positive Rate', size=24)
    plt.legend(loc='lower right', prop={'size': 18})

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.title('Mean ROC curves after 10-fold outer CV', size=28)

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
            sensitivity = 0
        else:
            sensitivity = true_positives / (true_positives + false_negatives)


        if 2 * true_positives + false_positives + false_negatives == 0:
            f1_score = 0
        else:
            f1_score = (2 * true_positives) / (2 * true_positives + false_positives + false_negatives)


        if (true_negatives + false_positives == 0) or (true_negatives + false_negatives == 0) or \
                (true_positives + false_positives == 0) or (true_positives + false_negatives == 0):
            MCC = 0
        else:
            MCC = (true_positives * true_negatives - false_positives * false_negatives) / \
                  (math.sqrt((true_positives + false_positives) *
                             (true_positives + false_negatives) *
                             (true_negatives + false_positives) * (true_negatives + false_negatives)))

        class_symbol_to_evaluation_metrics[class_symbol]['true_positives'] = true_positives
        class_symbol_to_evaluation_metrics[class_symbol]['false_positives'] = false_positives

        class_symbol_to_evaluation_metrics[class_symbol]['true_negatives'] = true_negatives
        class_symbol_to_evaluation_metrics[class_symbol]['false_negatives'] = false_negatives

        class_symbol_to_evaluation_metrics[class_symbol]['accuracy'] = accuracy
        class_symbol_to_evaluation_metrics[class_symbol]['precision'] = precision
        class_symbol_to_evaluation_metrics[class_symbol]['sensitivity'] = sensitivity
        class_symbol_to_evaluation_metrics[class_symbol]['f1_score'] = f1_score
        class_symbol_to_evaluation_metrics[class_symbol]['MCC'] = MCC

    return class_symbol_to_evaluation_metrics


def compute_average_performance_metrics_for_binary_classification(performance_metrics, cancer_data=True):

    keys = performance_metrics.keys()

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

    print "Average performance meterics"
    print average_performance_metrics

    print "Standard deviation"
    print std_performance_metrics

    perf = dict()

    perf['average'] = average_performance_metrics
    perf['std'] = std_performance_metrics


    return perf


def compute_performance_metrics_for_multiclass_classification(performance_metrics):
    keys = performance_metrics.keys()

    keys = performance_metrics.keys()

    average_performance_metric = dict()
    std_performance_metric = dict()

    accuracy = []
    precision = []
    sensitivity = []
    MCC = []
    f1_score = []

    for key in keys:
        fold_performance_metrics = performance_metrics[key]

        accuracy += [fold_performance_metrics['accuracy']]
        precision += [fold_performance_metrics['precision']]
        sensitivity += [fold_performance_metrics['sensitivity']]
        MCC += [fold_performance_metrics['MCC']]
        f1_score += [fold_performance_metrics['f1_score']]

    average_performance_metric['accuracy'] = np.mean(accuracy)
    average_performance_metric['precision'] = np.mean(precision)
    average_performance_metric['sensitivity'] = np.mean(sensitivity)
    average_performance_metric['MCC'] = np.mean(MCC)
    average_performance_metric['f1_score'] = np.mean(f1_score)

    std_performance_metric['accuracy'] = np.std(accuracy)
    std_performance_metric['precision'] = np.std(precision)
    std_performance_metric['sensitivity'] = np.std(sensitivity)
    std_performance_metric['MCC'] = np.std(MCC)
    std_performance_metric['f1_score'] = np.std(f1_score)

    print "average"
    print average_performance_metric
    print "standard deviation"
    print std_performance_metric

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


    return p_values


def paired_t_test_multiclass_classification(performance_metrics_first_model, performance_metrics_second_model):

    keys = performance_metrics_first_model.keys()

    p_values = dict()

    accuracy_first_model = []
    precision_first_model = []
    sensitivity_first_model = []
    MCC_first_model = []
    f1_score_first_model = []

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

        accuracy_first_model += [fold_performance_metrics_first_model['accuracy']]
        accuracy_second_model += [fold_performance_metrics_second_model['accuracy']]

        precision_first_model += [fold_performance_metrics_first_model['precision']]
        precision_second_model += [fold_performance_metrics_second_model['precision']]

        sensitivity_first_model += [fold_performance_metrics_first_model['sensitivity']]
        sensitivity_second_model += [fold_performance_metrics_second_model['sensitivity']]

        MCC_first_model += [fold_performance_metrics_first_model['MCC']]
        MCC_second_model += [fold_performance_metrics_second_model['MCC']]

        f1_score_first_model += [fold_performance_metrics_first_model['f1_score']]
        f1_score_second_model += [fold_performance_metrics_second_model['f1_score']]


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


    print "P values"
    print p_values

    return p_values


def compute_micro_average(class_symbol_to_performance_metrics):

    micro_average = dict()

    keys = class_symbol_to_performance_metrics.keys()

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for key in keys:
        true_positives += class_symbol_to_performance_metrics[key]['true_positives']
        false_positives += class_symbol_to_performance_metrics[key]['false_positives']
        true_negatives += class_symbol_to_performance_metrics[key]['true_negatives']
        false_negatives += class_symbol_to_performance_metrics[key]['false_negatives']

    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_negatives + false_positives)

    if true_positives + false_positives == 0:
        precision = 0
    else:
        precision = true_positives / (true_positives + false_positives)

    if true_positives + false_negatives == 0:
        sensitivity = 0
    else:
        sensitivity = true_positives / (true_positives + false_negatives)

    f1_score = (2 * true_positives) / (2 * true_positives + false_positives + false_negatives)

    if (true_negatives + false_positives == 0) or (true_negatives + false_negatives == 0) or \
            (true_positives + false_positives == 0) or (true_positives + false_negatives == 0):
        MCC = 0
    else:
        MCC = (true_positives * true_negatives - false_positives * false_negatives) / \
              (math.sqrt((true_positives + false_positives) * (true_positives + false_negatives) * \
                         (true_negatives + false_positives) * (true_negatives + false_negatives)))

    micro_average['accuracy'] = accuracy
    micro_average['precision'] = precision
    micro_average['sensitivity'] = sensitivity
    micro_average['f1_score'] = f1_score
    micro_average['MCC'] = MCC

    print "micro average"
    print micro_average

    return micro_average


def compute_macro_average(class_symbol_to_performance_metrics):

    keys = class_symbol_to_performance_metrics.keys()

    macro_performance_metrics = dict()

    accuracy = []
    precision = []
    sensitivity = []
    MCC = []
    f1_score = []

    for key in keys:

        if (class_symbol_to_performance_metrics[key]['true_positives'] == 0) and \
            (class_symbol_to_performance_metrics[key]['false_negatives'] == 0):
            print "Missing class " + str(key)
        else:
            accuracy += [class_symbol_to_performance_metrics[key]['accuracy']]
            precision += [class_symbol_to_performance_metrics[key]['precision']]
            sensitivity += [class_symbol_to_performance_metrics[key]['sensitivity']]
            MCC += [class_symbol_to_performance_metrics[key]['MCC']]
            f1_score += [class_symbol_to_performance_metrics[key]['f1_score']]


    macro_performance_metrics['accuracy'] = np.mean(accuracy)
    macro_performance_metrics['precision'] = np.mean(precision)
    macro_performance_metrics['sensitivity'] = np.mean(sensitivity)
    macro_performance_metrics['MCC'] = np.mean(MCC)
    macro_performance_metrics['f1_score'] = np.mean(f1_score)

    print "macro"
    print macro_performance_metrics

    return macro_performance_metrics

