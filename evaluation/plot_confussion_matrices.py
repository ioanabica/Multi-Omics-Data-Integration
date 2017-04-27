import pandas as pd
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
from evaluation_metrics import compute_class_id_to_class_symbol


sn.reset_orig()

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

    #fig = plt.figure(figsize=(8, 10), dpi=150)
    fig, ax = plt.subplots(figsize=(10, 15), dpi=150)
    sn.heatmap(df_cm, annot=True, ax=ax, cbar=False)

    ax.xaxis.set_ticks_position('top')
    #plt.xticks(rotation=30)
    plt.yticks(rotation=0)
    plt.title(plt_title, size=18, y=-0.08)

    plt.xlabel('Predicted Class', size=14)
    #ax.xaxis.set_label_position('top')
    ax.xaxis.set_label_coords(0.5, 1.1)
    plt.ylabel('Actual Class', size=14)
    plt.show()


def plot_confussion_matrix_as_heatmap_for_cancer_data(confussion_matrix, plt_title):

    confussion_matrix = np.array(confussion_matrix)

    labels = ['Not Cancer', 'Cancer']

    df_cm = pd.DataFrame(confussion_matrix, index=[i for i in labels], columns=[i for i in labels])

    fig = plt.figure(figsize=(3, 3), dpi=150)
    ax = plt.subplot(111)
    sn.heatmap(df_cm, annot=True, fmt='g')

    ax.xaxis.set_ticks_position('top')
    #plt.xticks(rotation=30)
    plt.yticks(rotation=0)
    plt.title(plt_title, size=18, y=-0.09)


    plt.xlabel('Predicted Class', size=16)
    ax.xaxis.set_label_position('top')
    plt.ylabel('Actual Class', size=16)
    plt.show()