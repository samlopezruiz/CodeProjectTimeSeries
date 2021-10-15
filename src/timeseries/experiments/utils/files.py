import copy
import os
from datetime import datetime
import joblib


def create_dir(file_path):
    if not isinstance(file_path, list):
        raise Exception('file path is not a list: {}'.format(file_path))

    for i in range(1, len(file_path)):
        path = os.path.join(*file_path[:i])
        if not os.path.exists(path):
            os.makedirs(path)


def save_df(df, file_path=['results', 'res'], use_date=False):
    create_dir(file_path)
    path = os.path.join(get_new_file_path(file_path, '.csv', use_date))
    print('Saving Dataframe to: \n{}'.format(path))
    df.to_csv(path)


def save_vars(vars, file_path=['results', 'res'], use_date_suffix=False):
    create_dir(file_path)
    path = os.path.join(*get_new_file_path(file_path, '.z', use_date_suffix))
    print('Saving Vars to: \n{}'.format(path))
    joblib.dump(vars, path)


def get_new_file_path(file_path, extension, use_date_suffix):
    ex = len(extension)
    new_file_path = copy.copy(file_path)
    if use_date_suffix:
        new_file_path[-1] = new_file_path[-1] + '_' + datetime.now().strftime("%d_%m_%Y %H-%M") + extension
    else:
        new_file_path[-1] = new_file_path[-1] + extension
        if os.path.exists(os.path.join(*new_file_path)):
            counter = 1
            new_file_path[-1] = '{}_1{}'.format(new_file_path[-1][:-ex], extension)
            while True:
                new_file_path[-1] = '{}{}{}'.format(new_file_path[-1][:-(ex + 1)],
                                                    str(counter),
                                                    extension)
                if not os.path.exists(os.path.join(*new_file_path)):
                    return new_file_path
                else:
                    counter += 1
        else:
            return new_file_path
    return new_file_path
