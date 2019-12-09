import pandas as pd
import numpy as np
from os import listdir, makedirs
from os.path import isfile, join, exists
from tqdm import tqdm
import operator


def create_segments_and_labels(dataframe, time_steps, step, n_features, label_class, label_2):

    # Down Sampling - Manual approach to solve data imbalance problem
    class_count_map = {}
    for i in range(0, len(dataframe) - time_steps, step):
        timestep_data = dataframe[label_class][i: i + time_steps]
        # If multiple classes in same segment, continue
        if len(set(timestep_data)) != 1:
            continue
        class_count_map[i] = timestep_data.iloc[0]

    sorted_class_count_map = sorted(class_count_map.items(), key=operator.itemgetter(1), reverse=False)

    segments = []
    labels = []
    regression_values = []
    for i, c in sorted_class_count_map:
        xs = dataframe['X'].values[i: i + time_steps]
        ys = dataframe['Y'].values[i: i + time_steps]
        zs = dataframe['Z'].values[i: i + time_steps]
        # Retrieve the most often used label in this segment
        class_label = c
        class_reg = dataframe[label_2][i: i + time_steps].mean()
        segments.append([xs, ys, zs])
        labels.append(class_label)
        regression_values.append(class_reg)

    # Bring the segments into a better shape
    reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, time_steps, n_features)
    labels = np.asarray(labels)
    regression_values = np.asarray(regression_values)

    return {'segments': reshaped_segments, 'activity_classes': labels, 'energy_e': regression_values}


if __name__ == "__main__":

    TIME_PERIODS_LIST = [6000]
    N_FEATURES = 3
    LABEL_CLASS = 'waist_intensity'
    LABEL_REG = 'waist_ee'

    req_cols = ['X', 'Y', 'Z', 'waist_ee', 'waist_intensity']
    input_cols = ['X', 'Y', 'Z']
    target_cols = ['waist_ee', 'waist_intensity']

    INPUT_DATA_FOLDER = "../data/raw/"
    OUTPUT_FOLDER_ROOT = '../data/test_ready/'

    all_files = [f for f in listdir(INPUT_DATA_FOLDER) if isfile(join(INPUT_DATA_FOLDER, f))]

    for f in tqdm(all_files):

        try:
            df = pd.read_csv(join(INPUT_DATA_FOLDER, f), usecols=req_cols)

            class_counts = df['waist_intensity'].value_counts(normalize=True)

            for time_window in TIME_PERIODS_LIST:

                # No overlap for test data
                STEP_DISTANCE = time_window

                reshaped_outcomes = create_segments_and_labels(df, time_window, STEP_DISTANCE,
                                                               N_FEATURES, LABEL_CLASS, LABEL_REG)

                OUTPUT_FOLDER = join(OUTPUT_FOLDER_ROOT,
                                     'window-{}-overlap-{}'.format(time_window, STEP_DISTANCE))
                if not exists(OUTPUT_FOLDER):
                    makedirs(OUTPUT_FOLDER)
                out_name = join(OUTPUT_FOLDER, f.replace('.csv', '_test_ready.npy'))
                np.save(out_name, reshaped_outcomes)

        except Exception as e:
            print('Error loading file {}.\nError: {}'.format(f, e))

    print('Completed.')
