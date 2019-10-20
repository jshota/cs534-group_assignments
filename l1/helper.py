import math
import numpy as np
import datetime

def normalize(matrix):
    xmins = np.min(matrix, axis=0)
    xmaxs = np.max(matrix, axis=0)

    for i in range(1, len(matrix[0])):
        for j in range(0, len(matrix)):
            matrix[j][i] =  (matrix[j][i] - xmins[i])/\
                            (xmaxs[i] - xmins[i])
    return matrix

def import_data(file_directory, is_normalize):
    x, y = [], []
    today = datetime.datetime.now()
    head = True

    with open(file_directory) as f:
        line_list = f.readlines()
        for line in line_list:
            # filter out header
            if head:
                head = False
                continue
            # csv data seperate by comma
            separate_line = line.split(',')

            # 0) b). split the date feature into three separate numerical features
            date = datetime.datetime.strptime(separate_line[2], '%m/%d/%Y')
            year = date.year
            month = date.month
            day = date.day
            past_days = (today.year - year)*365 + (today.month - month)*30 + today.day - day

            # create dataset
            y.append(float(separate_line.pop().replace('\n', ''))) # target value

            # set of features
            features = []
            for i in range(len(separate_line)):
                if i == 1:
                    # remove the 'id'
                    continue
                elif i == 2:
                    # use 'past days' instead of 'date'
                    features.append(past_days)
                else:
                    features.append(float(separate_line[i]))
            x.append(features)

        if is_normalize:
            return [normalize(x), y]
        else:
            return [x, y]