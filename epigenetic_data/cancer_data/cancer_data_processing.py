


def extract_patients_data(data_file):

    patient_id_to_gene_expressions = dict()
    patient_id_to_dna_methilation = dict()
    patient_id_to_gene_expressions_and_dna_methlation = dict()
    patient_id_to_label = dict()

    label_to_patient_ids = dict()
    label_to_patient_ids['cancer'] = list()
    label_to_patient_ids['not cancer'] = list()


    num_patients = int(data_file.readline())
    print num_patients
    data_file.readline()
    data_file.readline()

    for index in range(num_patients):
        patient_id = 'patient ' + str(index)

        label = data_file.readline().split()

        if label[0] =='patient':
            diagnosis_label = 'cancer'
            patient_id_to_label[patient_id] = diagnosis_label
            label_to_patient_ids[diagnosis_label].append(patient_id)
        else:
            diagnosis_label = 'not cancer'
            patient_id_to_label[patient_id] = diagnosis_label
            label_to_patient_ids[diagnosis_label].append(patient_id)

        gene_expression_levels = data_file.readline().split()
        patient_id_to_gene_expressions[patient_id] = gene_expression_levels

        dna_methylation = data_file.readline().split()
        patient_id_to_dna_methilation[patient_id] = dna_methylation

        patient_id_to_gene_expressions_and_dna_methlation[patient_id] = gene_expression_levels + dna_methylation

        data_file.readline()

    print patient_id_to_gene_expressions_and_dna_methlation

    return patient_id_to_gene_expressions, patient_id_to_dna_methilation, \
           patient_id_to_gene_expressions_and_dna_methlation, patient_id_to_label, label_to_patient_ids


def create_one_hot_encoding(labels):
    label_to_one_hot_encoding = dict()

    for index in range(len(labels)):
        one_hot_encoding = [0.0] * len(labels)
        one_hot_encoding[index] = 1.0
        label_to_one_hot_encoding[labels[index]] = one_hot_encoding

    return label_to_one_hot_encoding


def create_label_to_patient_ids(patient_ids, patient_id_to_label):
    label_to_patient_ids = dict()

    for patient_id in patient_ids:
        label = patient_id_to_label[patient_id]

        if label in label_to_patient_ids.keys():
            label_to_patient_ids[label] += [patient_id]
        else:
            label_to_patient_ids[label] = [patient_id]


    return label_to_patient_ids