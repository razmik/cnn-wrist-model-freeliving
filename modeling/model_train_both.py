from os import listdir
from os.path import join, isdir
import modeling.model_train_class as model_train_class
import modeling.model_train_reg as model_train_reg


if __name__ == '__main__':

    # Get folder names
    temp_folder = '../data/train_ready/'
    all_files = [f for f in listdir(temp_folder) if isdir(join(temp_folder, f))]

    allowed_windows = [6000]
    trial_num = 4

    # Train classification model
    for f in sorted(all_files, reverse=True):

        if int(f.split('-')[1]) not in allowed_windows:
            continue

        print('\n\n\n\nProcessing Classification {}'.format(f))
        model_train_class.run(f, trial_num)

    # Train regression model
    for f in sorted(all_files, reverse=True):

        if int(f.split('-')[1]) not in allowed_windows:
            continue

        print('\n\n\n\nProcessing Regression {}'.format(f))
        model_train_reg.run(f, trial_num)

    print('Completed')
