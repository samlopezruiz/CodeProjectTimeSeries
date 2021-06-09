import os


def new_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)