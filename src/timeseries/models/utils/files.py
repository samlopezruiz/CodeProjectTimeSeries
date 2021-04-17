import joblib
import os
import datetime


def save_vars(vars, file_path=None):

    if file_path is None:
        file_path = ['results', 'result']
    print("saving vars .z")
    if not os.path.exists(file_path[0]):
        os.makedirs(file_path[0])
    path = file_path[:-1].copy() + [file_path[-1] + '_' + datetime.datetime.now().strftime("%Y_%m_%d_%H-%M") + ".z"]
    joblib.dump(vars, os.path.join(*path))


if __name__ == '__main__':
    a = 1
    save_vars([a])
