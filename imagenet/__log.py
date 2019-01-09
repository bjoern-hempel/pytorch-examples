"""
Log functions for writing csv and log files

@author: Björn Hempel <bjoern@hempel.li>
@version: 1.0 (2019-01-04)
"""
import datetime
import os
import random
import time
import torch


"""

"""
def getFormatedPath(
    path=None,
    name='validated',
    epoch=0,
    time=None,
    learning_rate=0.001,
    momentum=0.9,
    batch_size=256,
    workers=4,
    weight_decay=0.0001,
    pretrained=True,
    file='{}_lr{}_m{}_bs{}_w{}_wd{}_{}.{}csv'
):
    file = file.replace('{}', name + '_e' + str(epoch) if epoch else name, 1)

    file_formated = file.format(
        learning_rate,
        momentum,
        batch_size,
        workers,
        weight_decay,
        'p' if pretrained else 'r',
        '' if time is None else datetime.datetime.fromtimestamp(time.timestamp()).strftime('%Y%m%d_%H%M%S') + '.'
    )

    return os.path.join(path, file_formated) if path is not None else file_formated


"""
Gets the csv handler

@author: Björn Hempel <bjoern@hempel.li>
@version: 1.0 (2019-01-04)
@param csv_file: the file handler from csv file
@param values: the given values will be writen to csv file
@return: null
"""
def getCSVHandler(csv_file, *values):

    # create folder for csv file
    createFolderForFile(csv_file)

    # move existing file to timestamp file
    if os.path.exists(csv_file):
        os.rename(csv_file, getDateName(csv_file))

    csv_handler = open(csv_file, 'a')
    writeCSV(csv_handler, True, *values)
    return csv_handler


"""

"""
def getCSVHandlerWithHeader(path, name, epoch, args, time):

    headers = {
        'settings': [
            'name', 'value'
        ],
        'summary': [
            'time taken', 'train session', 'epoch number current', 'epoch number overall',
            'current phase (train or val)', 'loss', 'accuracy in percent', 'best accuracy in percent',
            'learning rate'
        ],
        'summary_full': [
            'time taken', 'train session', 'epoch number current', 'epoch number overall',
            'current phase (train or val)', 'current processed images', 'correct processed images',
            'images overall', 'processed in percent', 'loss', 'accuracy in percent', 'learning rate'
        ],
        'validated': [
            'time taken', 'file name', 'status', 'real class', 'predicted class 1', 'predicted class 1 (%)',
            'predicted class 2', 'predicted class 2 (%)', 'predicted class 3', 'predicted class 3 (%)',
            'predicted class 4', 'predicted class 4 (%)', 'predicted class 5', 'predicted class 5 (%)'
        ]
    }

    # prepare summary csv
    csv_path = getFormatedPath(
        path,
        name,
        epoch,
        time,
        args.lr,
        args.momentum,
        args.batch_size,
        args.workers,
        args.weight_decay,
        args.pretrained
    )

    header = headers[name]

    return getCSVHandler(
        csv_path,
        *header
    )


"""
Gets the given file name with time

@author: Björn Hempel <bjoern@hempel.li>
@version: 1.0 (2019-01-05)
@param file_name: the file name without time
@return: the file name with time
"""
def getDateName(file_name):

    time = os.path.getmtime(file_name)

    date_str = datetime.datetime.fromtimestamp(
        int(float(time))
    ).strftime('%Y%m%d_%H%M%S')

    base = os.path.splitext(file_name)[0]
    ext = os.path.splitext(file_name)[1]

    return base + '.' + date_str + ext


"""
Return the given image an return the transformed version.

@author: Björn Hempel <bjoern@hempel.li>
@version: 1.0 (2018-12-01)
@param imgPath: the path to image
@return: the image in binary format
"""
def log(text, startTime, logFullFile, logFile, type = 'info'):
    # Todo: check type input parameter.

    text = '{}: {}'.format(
        datetime.datetime.now() - startTime,
        text
    )

    print(text)

    if type == 'info':
        return True

    # log to full log file
    logFullFile.write(text + '\n')
    logFullFile.flush()

    if type != 'log':
        return True

    # log to log file
    logFile.write(text + '\n')
    logFile.flush()


"""
A CSV write function

@author: Björn Hempel <bjoern@hempel.li>
@version: 1.0 (2018-12-01)
@param file: the file handler from csv file
@param values: the given values will be writen to csv file
@return: null
"""
def writeCSV(file, header, *values):

    if header:
        csvLine = '"timestamp"'
    else:
        csvLine = '{:.6f}'.format(
            time.time()
        )

    for value in values:
        valueStr = ''

        if type(value) is str:
            valueStr = '"' + value + '"'
        elif type(value) is int:
            valueStr = str(value)
        elif type(value) is float:
            valueStr = str(value)
        elif type(value) is bool:
            valueStr = 'True' if value else 'False'
        elif type(value) is torch.Tensor:
            valueStr = str(value.item())
        elif type(value) is None.__class__:
            valueStr = 'None'
        elif type(value) is list:
            #print('ignore list elements')
            continue
        elif type(value) is dict:
            #print('ignore dict elements')
            continue
        else:
            print(type(value))
            print(value)
            exit()

        if csvLine != '':
            csvLine += ','

        csvLine += valueStr

    file.write(csvLine + '\n')
    file.flush()


"""
Gets the next free file path.

@author  Björn Hempel <bjoern@hempel.li>
@version 1.0 (20181129)
@param path: the path to the file
@return: the next available path to file
"""
def getNextFile(path):
    fileVersionCounter = 0
    while True:
        filePath = path.format(str(fileVersionCounter).zfill(4))

        if not os.access(filePath, os.F_OK):
            return filePath

        fileVersionCounter += 1

"""
Gets all properties of given class.

@author  Björn Hempel <bjoern@hempel.li>
@version 1.0 (20190106)
@param cls: the class to determine
@return: array of the given class
"""
def getClassProperties(cls):
    return [i for i in cls.__dict__.keys() if i[:1] != '_']

"""

"""
def createFolderForFile(file, mode=0o775):
    path = os.path.dirname(file)
    os.makedirs(path, mode, True)
