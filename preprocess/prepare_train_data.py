import pandas as pd
import numpy as np
from os import listdir, makedirs
from os.path import isfile, join, exists
from tqdm import tqdm
from random import sample
import operator


def create_segments_and_labels(df, time_steps, step, n_features, label_class, label_2):

    # Down Sampling - Manual approach to solve data imbalance problem
    lmvpa_begin_class = 2
    class_count_map = {}
    for i in range(0, len(df) - time_steps, step):
        timestep_data = df[label_class][i: i + time_steps]
        # If multiple classes in same segment, continue
        if len(set(timestep_data)) != 1:
            continue
        class_count_map[i] = timestep_data.iloc[0]

    sorted_class_count_map = sorted(class_count_map.items(), key=operator.itemgetter(1), reverse=False)
    ordered_classes = [i for _, i in sorted_class_count_map]
    divide_index = ordered_classes.index(lmvpa_begin_class)
    lmvpa = sorted_class_count_map[divide_index:]
    sb = sample(sorted_class_count_map[:divide_index], len(lmvpa)) if divide_index > len(lmvpa) else sorted_class_count_map[:divide_index]
    filtered_tuples = sb + lmvpa

    segments = []
    labels = []
    regression_values = []
    for i, c in filtered_tuples:
        xs = df['X'].values[i: i + time_steps]
        ys = df['Y'].values[i: i + time_steps]
        zs = df['Z'].values[i: i + time_steps]
        # Retrieve the most often used label in this segment
        class_label = c
        class_reg = df[label_2][i: i + time_steps].mean()
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
    OUTPUT_FOLDER_ROOT = '../data/train_ready/'

    all_files = [f for f in listdir(INPUT_DATA_FOLDER) if isfile(join(INPUT_DATA_FOLDER, f))]
    for f in tqdm(all_files):

        try:
            df = pd.read_csv(join(INPUT_DATA_FOLDER, f), usecols=req_cols)

            class_counts = df['waist_intensity'].value_counts(normalize=True)
            if class_counts.shape[0] == 1:
                continue

            for time_window in TIME_PERIODS_LIST:

                # Set overlapping window to 1/2 of the epoch
                STEP_DISTANCE = int(time_window / 2)

                reshaped_outcomes = create_segments_and_labels(df, time_window, STEP_DISTANCE,
                                                               N_FEATURES, LABEL_CLASS, LABEL_REG)

                OUTPUT_FOLDER = join(OUTPUT_FOLDER_ROOT,
                                     'window-{}-overlap-{}'.format(time_window, STEP_DISTANCE))
                if not exists(OUTPUT_FOLDER):
                    makedirs(OUTPUT_FOLDER)
                out_name = join(OUTPUT_FOLDER, f.replace('.csv', '_train_ready.npy'))
                np.save(out_name, reshaped_outcomes)

        except Exception as e:
            print('Error loading file {}.\nError: {}'.format(f, e))

    print('Completed.')
