from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from os import listdir, makedirs
from os.path import join, isfile, exists
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shutil
from tqdm import tqdm
from time import time
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Reshape, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.callbacks import TensorBoard

from scipy.stats.stats import pearsonr
from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score, confusion_matrix
from math import sqrt
import statistical_extensions as SE

pd.options.display.float_format = '{:.1f}'.format
sns.set()  # Default seaborn look and feel
plt.style.use('ggplot')
print('Keras version ', keras.__version__)


def load_data(filenames):

    X_data = []
    Y_data = []
    ID_user = []
    for filename in tqdm(filenames):
        npy = np.load(filename, allow_pickle=True)
        X_data.append(npy.item().get('segments'))
        Y_data.append(npy.item().get('energy_e'))

        user_id = filename.split('/')[-1][:6]
        data_length = npy.item().get('energy_e').shape[0]
        ID_user.extend([user_id for _ in range(data_length)])

    X_data = np.concatenate(X_data, axis=0)
    Y_data = np.concatenate(Y_data, axis=0)

    return X_data, Y_data, ID_user


def plot_model(history, MODEL_FOLDER):
    # summarize history for accuracy and loss
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['acc'], "g--", label="Accuracy of training data")
    plt.plot(history.history['val_acc'], "g", label="Accuracy of validation data")
    plt.plot(history.history['loss'], "r--", label="Loss of training data")
    plt.plot(history.history['val_loss'], "r", label="Loss of validation data")
    plt.title('Model Accuracy and Loss')
    plt.ylabel('Accuracy and Loss')
    plt.xlabel('Training Epoch')
    plt.ylim(0)
    plt.legend()
    plt.savefig(MODEL_FOLDER + 'learning_history.png')
    plt.clf()
    plt.close()


def run(FOLDER_NAME, trial_id):

    DATA_ROOT = '../data/train_ready/'
    TRAIN_DATA_FOLDER = DATA_ROOT + '/{}/'.format(FOLDER_NAME)
    TEST_DATA_FOLDER = DATA_ROOT + '/{}/'.format(FOLDER_NAME)
    OUTPUT_FOLDER_ROOT = '../output/regression/v{}/{}'.format(trial_id, FOLDER_NAME)

    MODEL_FOLDER = OUTPUT_FOLDER_ROOT + '/model_out/'
    RESULTS_FOLDER = OUTPUT_FOLDER_ROOT + '/results/'
    if not exists(OUTPUT_FOLDER_ROOT):
        makedirs(OUTPUT_FOLDER_ROOT)
        makedirs(MODEL_FOLDER)
        makedirs(RESULTS_FOLDER)

    # Create temp folder to save model outputs
    temp_model_out_folder = 'temp_model_out'
    makedirs(temp_model_out_folder)

    results_descriptions = []

    # The number of steps within one time segment
    TIME_PERIODS = int(FOLDER_NAME.split('-')[1])
    # The steps to take from one segment to the next; if this value is equal to
    # TIME_PERIODS, then there is no overlap between the segments
    STEP_DISTANCE = int(FOLDER_NAME.split('-')[3])
    LABEL = 'energy_expenditure'
    results_descriptions.append('Time Period = {}, Step Distance = {}, Label = {}'.format(TIME_PERIODS, STEP_DISTANCE, LABEL))

    # Load all data
    all_files_train = [join(TRAIN_DATA_FOLDER, f) for f in listdir(TRAIN_DATA_FOLDER) if
                       isfile(join(TRAIN_DATA_FOLDER, f))]
    all_files_test = [join(TEST_DATA_FOLDER, f) for f in listdir(TEST_DATA_FOLDER) if isfile(join(TEST_DATA_FOLDER, f))]
    results_descriptions.append('Train files: {}\nTest files: {}'.format(len(all_files_train), len(all_files_test)))

    train_X_data, train_Y_data, train_ID_user = load_data(all_files_train)
    test_X_data, test_Y_data, test_ID_user = load_data(all_files_test)

    assert train_X_data.shape[0] == train_Y_data.shape[0] == len(train_ID_user)
    assert test_X_data.shape[0] == test_Y_data.shape[0] == len(test_ID_user)

    results_descriptions.append('Subjects in Train set = {}'.format(len(set(train_ID_user))))
    results_descriptions.append('Subjects in Test set = {}'.format(len(set(test_ID_user))))
    results_descriptions.append('All unique users = {}'.format(len(set(train_ID_user + test_ID_user))))

    # Train /Test split

    X_train, X_test = train_X_data, test_X_data
    y_train, y_test = train_Y_data, test_Y_data
    ID_train, ID_test = train_ID_user, test_ID_user

    # Data -> Model ready
    num_time_periods, num_sensors = X_train.shape[1], X_train.shape[2]

    # Set input_shape / reshape for Keras
    # Remark: acceleration data is concatenated in one array in order to feed
    # it properly into coreml later, the preferred matrix of shape [40,3]
    input_shape = (num_time_periods * num_sensors)
    X_train = X_train.reshape(X_train.shape[0], input_shape)
    X_test = X_test.reshape(X_test.shape[0], input_shape)

    # Convert type for Keras otherwise Keras cannot process the data
    X_train = X_train.astype("float32")
    y_train = y_train.astype("float32")
    X_test = X_test.astype("float32")
    y_test = y_test.astype("float32")

    # New architecture
    model_m = Sequential()
    model_m.add(Reshape((TIME_PERIODS, num_sensors), input_shape=(input_shape,)))
    model_m.add(Conv1D(80, 10, activation='relu', input_shape=(TIME_PERIODS, num_sensors)))
    model_m.add(Conv1D(100, 10, activation='relu'))
    model_m.add(MaxPooling1D(3))
    model_m.add(Conv1D(160, 10, activation='relu'))
    model_m.add(Conv1D(180, 10, activation='relu'))
    model_m.add(MaxPooling1D(3))
    model_m.add(Conv1D(220, 10, activation='relu'))
    model_m.add(Conv1D(240, 10, activation='relu'))
    model_m.add(GlobalMaxPooling1D())
    model_m.add(Dropout(0.5))
    model_m.add(Dense(1, activation='linear'))

    callbacks_list = [
        ModelCheckpoint(
            filepath='temp_model_out/best_model.{epoch:03d}-{val_loss:.2f}.h5',
            monitor='val_loss', save_best_only=True),
        TensorBoard(log_dir='logs/{}'.format(time())),
        EarlyStopping(monitor='val_loss', patience=15)
    ]

    model_m.compile(loss='mean_squared_error',
                    optimizer='adam',
                    metrics=['accuracy'])

    # Hyper-parameters
    BATCH_SIZE = 32
    EPOCHS = 20

    history = model_m.fit(X_train,
                          y_train,
                          batch_size=BATCH_SIZE,
                          epochs=EPOCHS,
                          callbacks=callbacks_list,
                          validation_split=0.2,
                          verbose=2)

    plot_model(history, MODEL_FOLDER)

    print('Selecting best model.')
    model_files = [(join(temp_model_out_folder, f), int(f.split('-')[0].split('.')[1])) for f in listdir(temp_model_out_folder) if
                   isfile(join(temp_model_out_folder, f)) and f.split('-')[0].split('.')[0] != 'final']

    model_b_name = sorted(model_files, key=lambda x: x[1], reverse=True)[0][0]
    results_descriptions.append('Best model name {}'.format(model_b_name))

    model_b = load_model(model_b_name)
    model_b.save(join(MODEL_FOLDER, model_b_name.split('\\')[1]))
    shutil.rmtree(temp_model_out_folder)

    # Evaluate against test data
    print('Model Evaluation.')
    y_pred_test = model_b.predict(X_test)

    assert y_test.shape[0] == y_pred_test.shape[0]

    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred_test)
    plt.xlabel('Actual EE')
    plt.ylabel('Predicted EE')
    plt.savefig(join(RESULTS_FOLDER, 'actual_vs_predicted_met.png'))
    plt.clf()
    plt.close()

    y_pred_test_1d_list = [list(i)[0] for i in list(y_pred_test)]
    corr = pearsonr(list(y_test), y_pred_test_1d_list)

    results_descriptions.append('\n\n -------RESULTS-------\n\n')
    results_descriptions.append('Pearsons Correlation = {}'.format(corr))
    results_descriptions.append('MSE - {}'.format(mean_squared_error(y_test, y_pred_test)))
    results_descriptions.append('RMSE - {}'.format(np.sqrt(mean_squared_error(y_test, y_pred_test))))
    results_descriptions.append('RMSE - {}'.format(sqrt(mean_squared_error(y_test, y_pred_test))))
    results_descriptions.append('R2 Error - {}'.format(r2_score(y_test, y_pred_test)))
    results_descriptions.append('Explained Variance Score - {}'.format(explained_variance_score(y_test, y_pred_test)))

    class_names = ['SED', 'LPA', 'MVPA']
    y_test_ai = SE.EnergyTransform.met_to_intensity(y_test)
    y_pred_test_ai = SE.EnergyTransform.met_to_intensity(y_pred_test)

    cnf_matrix = confusion_matrix(y_test_ai, y_pred_test_ai)

    stats = SE.GeneralStats.evaluation_statistics(cnf_matrix)

    assessment_result = 'Classes' + '\t' + str(class_names) + '\t' + '\n'
    assessment_result += 'Accuracy' + '\t' + str(stats['accuracy']) + '\t' + str(stats['accuracy_ci']) + '\n'
    assessment_result += 'Sensitivity' + '\t' + str(stats['sensitivity']) + '\n'
    assessment_result += 'Sensitivity CI' + '\t' + str(stats['sensitivity_ci']) + '\n'
    assessment_result += 'Specificity' + '\t' + str(stats['specificity']) + '\n'
    assessment_result += 'Specificity CI' + '\t' + str(stats['specificity_ci']) + '\n'

    results_descriptions.append(assessment_result)

    SE.GeneralStats.plot_confusion_matrix(cnf_matrix, classes=class_names, title='CM',
                                          output_filename=join(RESULTS_FOLDER, 'confusion_matrix.png'))

    results_df = pd.DataFrame(
        {'subject': ID_test,
         'waist_ee': list(y_test),
         'predicted_ee': [list(i)[0] for i in list(y_pred_test)]
         })

    def clean_data_points(data):
        data = data.assign(waist_ee_cleaned=data['waist_ee'])
        data = data.assign(predicted_ee_cleaned=data['predicted_ee'])
        data.loc[(data['predicted_ee'] < 1), 'predicted_ee_cleaned'] = 1
        return data

    results_df = clean_data_points(results_df)

    SE.BlandAltman.bland_altman_paired_plot_tested(results_df, '{}'.format(FOLDER_NAME), 1, log_transformed=True,
                                                   min_count_regularise=False, output_filename=join(RESULTS_FOLDER, 'bland_altman'))

    result_string = '\n'.join(results_descriptions)
    with open(join(RESULTS_FOLDER, 'result_report.txt'), "w") as text_file:
        text_file.write(result_string)


if __name__ == '__main__':

    # Get folder names
    temp_folder = '../data/train_ready/'
    all_files = [f for f in listdir(temp_folder) if os.path.isdir(join(temp_folder, f))]

    allowed_windows = [6000]
    trial_num = 3

    for f in all_files:

        if int(f.split('-')[1]) not in allowed_windows:
            continue

        print('\n\n\n\nProcessing {}'.format(f))
        run(f, trial_num)

