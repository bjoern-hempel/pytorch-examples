import os
import glob
import re
import pprint
import csv


# pretty printer
pp = pprint.PrettyPrinter(indent=4)

def is_float(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


def convert_data(data):
    if isinstance(data, str):
        if data.isdigit():
            return int(data)

        if is_float(data):
            return float(data)

        if data == 'False':
            return False

        if data == 'True':
            return True

    return data

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


def read_settings_csv(path_csv_settings):
    settings = {}
    counter = 0
    with open(path_csv_settings, newline='') as csvfile:
        counter += 1

        if counter <= 1:
            next(csvfile)

        data = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL, skipinitialspace=True)
        for row in data:
            settings[row[1]] = convert_data(row[2])

    return settings


def analyseArgs(args):
    if args.evaluate:
        # if evalutate was choosen, resume is needed
        if not args.resume:
            print('Evaluation without resume is not supported yet.')
            exit()

        # get csv settings config path
        path_csv_settings = get_settings_csv_from_model(args.resume)

        # get settings from csv settings
        settings = read_settings_csv(path_csv_settings)

        # The following elements are taken from the settings file
        takeovers = ['arch', 'session_name', 'linear_layer', 'epochs', 'learning_rate_decrease_after',
                     'learning_rate_decrease_factor', 'weight_decay', 'workers', 'momentum', 'pretrained', 'lr',
                     'model_path']

        # set some defaults from csv settings
        args.batch_size = 1
        for takeover in takeovers:
            setattr(args, takeover, settings[takeover])

        # set validated path if auto was given
        if args.csv_path_validated == 'auto':
            args.csv_path_validated = os.path.dirname(path_csv_settings)

    return args