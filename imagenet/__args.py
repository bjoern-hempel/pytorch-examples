import os
import glob
import re


def get_root_path(path):
    regex = re.compile(r'(.*)/models/(.*)', re.IGNORECASE)

    match = regex.search(path)

    if match is None:
        return None

    return match[1]


def get_csv_path(path):
    path_root = get_root_path(path)

    if path_root is None:
        return os.path.join(path, 'csv')

    return os.path.join(path_root, 'csv')


def get_csv_template(path):
    return os.path.basename(path).replace('model_best_', '{}_').replace('.pth', '.csv')


def get_settings_csv_from_model(path_model):
    path_root = get_root_path(path_model)

    if path_root is None:
        return None

    basename_csv_template = get_csv_template(path_model)

    for file_csv in glob.iglob('{}/**/{}'.format(path_root, basename_csv_template.format('settings')), recursive=True):
        return file_csv

    return None


def get_validated_path_from_model(path_model):
    file_csv_settings = get_settings_csv_from_model(path_model)

    if file_csv_settings is None:
        print('No settings csv found from model path "{}"'.format(path_model))

    return file_csv_settings.replace('settings_', 'validated_')


def analyseArgs(args):
    # if evalutate was choosen, resume is needed
    if args.evaluate:
        if not args.resume:
            print('Evaluation without resume is not supported yet.')
            exit()

    return args