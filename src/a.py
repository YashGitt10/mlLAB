import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, average_precision_score
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Input, Embedding, Lambda, concatenate, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adamax, Nadam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold



# Function to save best result
def save_func(file_path, values):
    file = [i.rstrip().split(',') for i in open(file_path).readlines()]
    file.append(values)
    file = pd.DataFrame(file)
    file.to_csv(file_path, header=None, index=None)

# Sigmoid Results to Binary    
def sigmoid_to_binary(predicted_labels):
    return [1 if i > 0.5 else 0 for i in predicted_labels]

# Create One Hot Layer
def OneHot(input_dim=None, input_length=None):
    def one_hot(x, num_classes):
        return K.one_hot(K.cast(x, 'uint8'), num_classes=num_classes)
    return Lambda(one_hot, arguments={'num_classes': input_dim}, input_shape=(input_length,))

# Data Conversion using Dictionary
def data_conversion(dataset, dictionary, max_length):
    matrix = np.zeros([len(dataset), max_length])
    for i in range(len(dataset)):
        dataset[i][1] = list(dataset[i][1])
        for j in range(len(dataset[i][1])):
            for k in dictionary:
                if dataset[i][1][j] == k:
                    dataset[i][1][j] = dictionary.get(k)
        if len(dataset[i][1]) < max_length:
            matrix[i, 0:len(dataset[i][1])] = dataset[i][1]
        else:
            matrix[i, 0:max_length] = dataset[i][1][0:max_length]
    return matrix.astype('int32')

# Generate Convolutional Layers
def generate_cov1D(num_filters, filter_window, stride, padding_method, act_func):
    return Conv1D(filters=num_filters, kernel_size=filter_window, strides=stride, padding=padding_method, activation=act_func)

# Generate Fully Connect Layers
def generate_fc(num_neurons, act_func):
    return Dense(units=num_neurons, activation=act_func)

# Generate embedding layers (needs to be the first layer)
def generate_embedding(input_dim, output_dim, input_len):
    return Embedding(input_dim=input_dim + 1, output_dim=output_dim, input_length=input_len)

# Transforms data to tensors
def generate_input(shape_size, dtype):
    return Input(shape=(shape_size,), dtype=dtype)

# Classifier Metrics
def f1_score(y_true, y_pred):
    y_pred = K.round(y_pred)
    precision = tf.keras.metrics.Precision()(y_true, y_pred)
    recall = tf.keras.metrics.Recall()(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

# Metrics Function: Sensitivity, Specificity, F1-Score, Accuracy and AUC
def metrics_function(sensitivity, specificity, f1, accuracy, auc_value, auprc_value, binary_labels, predicted_labels, labels_test, confusion_matrix):
    sensitivity_value = confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[1, 0])
    specificity_value = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[0, 1])
    precision = confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[0, 1])
    f1_value = 2 * (precision * sensitivity_value) / (precision + sensitivity_value)
    accuracy_value = accuracy_score(labels_test, np.array(binary_labels))
    auc = roc_auc_score(labels_test, predicted_labels)
    auprc = average_precision_score(labels_test, predicted_labels)
    metrics = []
    if sensitivity:
        metrics.append('Sensitivity:' + str(sensitivity_value))
    if specificity:
        metrics.append('Specificity:' + str(specificity_value))
    if f1:
        metrics.append('F1_Score:' + str(f1_value))
    if accuracy:
        metrics.append('Accuracy:' + str(accuracy_value))
    if auc_value:
        metrics.append('AUC:' + str(auc))
    if auprc_value:
        metrics.append('AUPRC: ' + str(auprc))
    return metrics






def generate_input(input_length, dtype):
    return Input(shape=(input_length,), dtype=dtype)

def generate_cov1D(filters, kernel_size, strides, padding, activation):
    return Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation)

def generate_fc(neurons, activation):
    return Dense(units=neurons, activation=activation)

def data_conversion(data, dictionary, seq_len):
    return [[dictionary.get(token, 0) for token in seq[:seq_len]] + [0]*(seq_len - len(seq)) for seq in data]

def sigmoid_to_binary(predicted_labels):
    return (predicted_labels > 0.5).astype(int)

def metrics_function(sensitivity, specificity, f1_score, accuracy, *args):
    # Assuming the metric functions return a string output
    return [f"Sensitivity:{sensitivity}", f"Specificity:{specificity}", f"F1_Score:{f1_score}", f"Accuracy:{accuracy}"]

def save_func(path, data):
    with open(path, 'a') as file:
        file.write(','.join(map(str, data)) + '\n')

def cnn_classifier(prot_data, smile_data, labels, prot_val, smile_val, labels_val, prot_seq_len, smile_len, encoding_type,
                   prot_dict_size, embedding_size, smile_dict_size, number_cov_layers, num_filters,
                   prot_filter_1, prot_filter_2, prot_filter_3, prot_filter_4, prot_filter_5,
                   act_func_conv, smile_filter_1, smile_filter_2, smile_filter_3, smile_filter_4, smile_filter_5,
                   number_fc_layers, fc_neurons_1, fc_neurons_2, fc_neurons_3, fc_neurons_4, fc_act_func, drop_rate,
                   output_act, optimizer_func, loss_func, metric_type, batch, epochs, option_validation, path):

    # Inputs
    protein_input = generate_input(prot_seq_len, 'int32')
    smile_input = generate_input(smile_len, 'int32')

    # Encoding Type
    if encoding_type == 'embedding':
        protein_embedding = Embedding(input_dim=prot_dict_size+1, output_dim=embedding_size)(protein_input)
        smile_embedding = Embedding(input_dim=smile_dict_size+1, output_dim=embedding_size)(smile_input)
    elif encoding_type == 'one_hot':
        protein_embedding = tf.one_hot(protein_input, depth=prot_dict_size)
        smile_embedding = tf.one_hot(smile_input, depth=smile_dict_size)

    # Convolutional Layers
    def apply_cnn_layers(embedding, filters, conv_layers, filter_sizes):
        for i in range(conv_layers):
            embedding = generate_cov1D(filters[i], filter_sizes[i], 1, 'valid', act_func_conv)(embedding)
        return embedding
    
    protein_cnn = apply_cnn_layers(protein_embedding, [num_filters, num_filters*2, num_filters*3, num_filters*4, num_filters*5][:number_cov_layers],
                                   number_cov_layers, [prot_filter_1, prot_filter_2, prot_filter_3, prot_filter_4, prot_filter_5][:number_cov_layers])
    
    smile_cnn = apply_cnn_layers(smile_embedding, [num_filters, num_filters*2, num_filters*3, num_filters*4, num_filters*5][:number_cov_layers],
                                 number_cov_layers, [smile_filter_1, smile_filter_2, smile_filter_3, smile_filter_4, smile_filter_5][:number_cov_layers])

    # Pooling Layers
    protein_pool = GlobalMaxPooling1D()(protein_cnn)
    smile_pool = GlobalMaxPooling1D()(smile_cnn)

    # Merge features
    features = concatenate([protein_pool, smile_pool])

    # Fully Connected Layers
    def apply_fc_layers(fc_layers, fc_neurons, features):
        for i in range(fc_layers):
            features = generate_fc(fc_neurons[i], fc_act_func)(features)
            if i < fc_layers - 1:  # Apply Dropout between FC layers
                features = Dropout(rate=drop_rate)(features)
        return features

    fc_layer = apply_fc_layers(number_fc_layers, [fc_neurons_1, fc_neurons_2, fc_neurons_3, fc_neurons_4][:number_fc_layers], features)

    # Output Layer
    output = generate_fc(1, output_act)(fc_layer)

    # Model
    model = Model(inputs=[protein_input, smile_input], outputs=output)
    model.compile(optimizer=optimizer_func, loss=loss_func, metrics=metric_type)

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_f1_score', patience=50, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(filepath=path.replace('.h5', ''), monitor='val_f1_score', save_best_only=True)

    # model_checkpoint = ModelCheckpoint(filepath=path, monitor='val_f1_score', save_best_only=True)

    if option_validation:
        model.fit([prot_data, smile_data], labels, batch_size=batch, epochs=epochs, validation_data=([prot_val, smile_val], labels_val),
                  callbacks=[early_stopping, model_checkpoint], class_weight={0: 0.36, 1: 0.64})
    else:
        model.fit([prot_data, smile_data], labels, batch_size=batch, epochs=epochs)

    return model


def grid_search(prot_train, smile_train, labels_train, prot_test, smile_test, labels_test, number_cov_layers, number_fc_layers,
                prot_seq_len, smile_len, prot_dict_size, smile_dict_size, encoding_type, embedding_size, num_filters, drop_rate,
                batch, learning_rate, prot_filter_1_window, prot_filter_2_window, prot_filter_3_window, prot_filter_4_window,
                prot_filter_5_window, smile_filter_1_window, smile_filter_2_window, smile_filter_3_window, smile_filter_4_window,
                smile_filter_5_window, fc_1_size, fc_2_size, fc_3_size, fc_4_size, act_func_conv, fc_act_func, epochs, loss_func, output_act, metric_type):

    for n_filter in num_filters:
        for d_rate in drop_rate:
            for l_rate in learning_rate:
                for smile_filter_1 in smile_filter_1_window:
                    for smile_filter_2 in smile_filter_2_window:
                        for smile_filter_3 in smile_filter_3_window:
                            for prot_filter_1 in prot_filter_1_window:
                                for prot_filter_2 in prot_filter_2_window:
                                    for prot_filter_3 in prot_filter_3_window:
                                        for fc_neurons_1 in fc_1_size:
                                            for fc_neurons_2 in fc_2_size:
                                                for fc_neurons_3 in fc_3_size:

                                                    file_name = f"{n_filter}_{d_rate}_{l_rate}_{smile_filter_1}_{smile_filter_2}_{smile_filter_3}_{prot_filter_1}_{prot_filter_2}_{prot_filter_3}_{fc_neurons_1}_{fc_neurons_2}_{fc_neurons_3}.keras"
                                                    path = f"./Models_CNN_FCNN/{file_name}.h5"
                                                    
                                                    model = cnn_classifier(prot_train, smile_train, labels_train, prot_test, smile_test, labels_test, prot_seq_len,
                                                                           smile_len, encoding_type, prot_dict_size, 0, smile_dict_size, number_cov_layers, n_filter,
                                                                           prot_filter_1, prot_filter_2, prot_filter_3, 0, 0, act_func_conv, smile_filter_1, smile_filter_2,
                                                                           smile_filter_3, 0, 0, number_fc_layers, fc_neurons_1, fc_neurons_2, fc_neurons_3, 0, fc_act_func,
                                                                           d_rate, output_act, Adam(learning_rate=l_rate), loss_func, metric_type, batch, epochs, True, path)
                                                    
                                                    predicted_labels = model.predict([prot_test, smile_test])
                                                    binary_labels = sigmoid_to_binary(predicted_labels)
                                                    cm = confusion_matrix(labels_test, binary_labels)
                                                    metric_values = metrics_function(True, True, True, True, False, False, binary_labels, predicted_labels, labels_test, cm)

                                                    save_func('../Results_CNN_FCNN.csv', [n_filter, d_rate, batch, l_rate, smile_filter_1, smile_filter_2, smile_filter_3,
                                                    prot_filter_1, prot_filter_2, prot_filter_3, fc_neurons_1, fc_neurons_2, fc_neurons_3, epochs, loss_func, output_act,
                                                    act_func_conv, fc_act_func, metric_values[0].strip('Sensitivity:'), metric_values[1].strip('Specificity:'),
                                                    metric_values[2].strip('F1_Score:'), metric_values[3].strip('Accuracy:')])


# Main block (example)
if __name__ == '__main__':
    prot_train = [i.rstrip().split(',') for i in open('./Datasets/Protein_Train_Dataset.csv')]
    prot_test = [i.rstrip().split(',') for i in open('./Datasets/Protein_Test_Dataset.csv')]
    drug_train = [i.rstrip().split(',') for i in open('./Datasets/Smile_Train_Dataset.csv')]
    drug_test = [i.rstrip().split(',') for i in open('./Datasets/Smile_Test_Dataset.csv')]

    prot_dictionary = json.load(open('./Dictionaries/aa_properties_dictionary.txt'))
    smile_dictionary = json.load(open('./Dictionaries/smile_dictionary.txt'))

    prot_train_data = data_conversion(prot_train, prot_dictionary, 1205)
    prot_test_data = data_conversion(prot_test, prot_dictionary, 1205)
    smile_train_data = data_conversion(drug_train, smile_dictionary, 90)
    smile_test_data = data_conversion(drug_test, smile_dictionary, 90)

    # Labels (example data loading)
    labels_train = np.load('./Labels/labels_train.npy')
    labels_test = np.load('./Labels/labels_test.npy')

    # labels_train = np.loadtxt('./Datasets/Labels_Train.csv', delimiter=',')
    # labels_test = np.loadtxt('./Datasets/Labels_Test.csv', delimiter=',')

    # Grid Search (example parameters)
    grid_search(prot_train_data, smile_train_data, labels_train, prot_test_data, smile_test_data, labels_test, number_cov_layers=3, number_fc_layers=3,
                prot_seq_len=1205, smile_len=90, prot_dict_size=len(prot_dictionary), smile_dict_size=len(smile_dictionary), encoding_type='embedding',
                embedding_size=[100], num_filters=[32], drop_rate=[0.5], batch=32, learning_rate=[0.001], prot_filter_1_window=[5], prot_filter_2_window=[7],
                prot_filter_3_window=[9], prot_filter_4_window=[0], prot_filter_5_window=[0], smile_filter_1_window=[5], smile_filter_2_window=[7],
                smile_filter_3_window=[9], smile_filter_4_window=[0], smile_filter_5_window=[0], fc_1_size=[128], fc_2_size=[64], fc_3_size=[32], fc_4_size=[0],
                act_func_conv='relu', fc_act_func='relu', epochs=50, loss_func='binary_crossentropy', output_act='sigmoid', metric_type=['accuracy'])
    
















