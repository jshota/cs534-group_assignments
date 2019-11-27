import numpy as np
import matplotlib.pyplot as plt


def read(file):
    data = np.genfromtxt(file, dtype=np.str, delimiter=',')
    # digit_class = []
    # if label:
    #     digit_class = data[:, -1]
    #     data = np.delete(data, -1, axis=1)
    '''
    for i in range(len(data[0]) - 1):
        if data[0][i] == 'veil-type_p':
            data = np.delete(data, i, axis=1)
    '''
    lable = data[0]
    data = np.delete(data, 0, axis=0)
    return data


def gini_index(data):
    kind, count = np.unique(data, return_counts=True)
    map = dict(zip(kind, count))
    uncertainty = 1
    for feature in map:
        prob_feature = map[feature] / len(data)
        uncertainty -= prob_feature ** 2
    return uncertainty


def benefit(left, right, gini):
    p = float(len(left)) / (len(left) + len(right))
    return gini - p * gini_index(left) - (1 - p) * gini_index(right)


def pick_features(data):
    best_gain = 0
    best_feature = 0
    feature_size = len(data[0])
    class_uncertainty = gini_index(data[:, -1])
    for i in range(feature_size - 1):
        y_label = data[:, -1]
        feature_i = data[:, i]
        left = np.array([])
        right = np.array([])
        for j in range(len(feature_i)):
            if feature_i[j] == '0':
                left = np.append(left, y_label[j])
            else:
                right = np.append(right, y_label[j])
        gain = benefit(left, right, class_uncertainty)
        if gain > best_gain:
            best_gain = gain
            best_feature = i
    return best_gain, best_feature


class Node(object):
    def __init__(self, data, feature):
        self.data = data
        self.feature = feature
        self.positive_child = None
        self.negative_child = None
        kind, count = np.unique(data[:, -1], return_counts=True)
        map = dict(zip(kind, count))
        max_label = None
        max_count = 0
        for lable in map:
            if map[lable] > max_count:
                max_label = lable
                max_count = map[lable]
        self.label = max_label


class Leaf(object):
    def __init__(self, data):
        kind, count = np.unique(data[:, -1], return_counts=True)
        map = dict(zip(kind, count))
        max_label = None
        max_count = 0
        for lable in map:
            if map[lable] > max_count:
                max_label = lable
                max_count = map[lable]
        self.label = max_label


def partition(data, feature_idx):
    feature_i = data[:, feature_idx]
    positive = np.array([])
    negative = np.array([])
    for j in range(len(feature_i)):
        if feature_i[j] == '0':
            if len(negative) == 0:
                negative = data[j]
            else:
                negative = np.row_stack((negative, data[j]))
        else:
            if len(positive) == 0:
                positive = data[j]
            else:
                positive = np.row_stack((positive, data[j]))
    return positive, negative


def build_tree(data, height, depth=2):
    best_gain, best_feature = pick_features(data)
    if best_gain == 0 or height >= depth:
        leaf = Leaf(data)
        return leaf
    positive, negative = partition(data, best_feature)
    node = Node(data, best_feature)
    if len(positive) > 0:
        node.positive_child = build_tree(positive, height + 1, depth)
    if len(negative) > 0:
        node.negative_child = build_tree(negative, height + 1, depth)
    return node


def classify(row, node, depth, max_depth):
    if node is None:
        return None
    if isinstance(node, Leaf):
        return node.label
    if depth >= max_depth:
        return node.label
    feature = node.feature
    if row[feature] == "1":
        return classify(row, node.positive_child, depth + 1, max_depth)
    else:
        return classify(row, node.negative_child, depth + 1, max_depth)


#   accuracy of decision tree
def validation(data, root, max_depth=2):
    error_count = 0
    for i in range(len(data)):
        label = classify(data[i], root, 0, max_depth)  # max_depth
        if label != data[i][-1]:
            error_count += 1
    k = error_count
    return 1 - float(error_count / len(data))


def decision_tree(depth):
    data_train = read("pa3_train.csv")
    data_valid = read("pa3_val.csv")
    dt_root = build_tree(data_train, 0, depth)
    print("accuracy on the train set:")
    accr_train = []
    accr_valid = []

    for dep in range(depth):
        dep += 1
        acc = validation(data_train, dt_root, dep)
        accr_train.append(acc)
        print("depth %d, accuracy %f" % (dep, acc))

    print("accuracy on the validation set:")
    for dep in range(depth):
        dep += 1
        acc = validation(data_valid, dt_root, dep)
        accr_valid.append(acc)
        print("depth %d, accuracy %f" % (dep, acc))

    it = np.linspace(1, depth, depth)
    plt.plot(it, accr_train, 'b', label='train data')
    plt.plot(it, accr_valid, 'r', label='valid data')
    plt.xlabel('depth')
    plt.ylabel('accuracies')
    plt.title('decisionTree: accuracy')
    plt.legend(loc=0)
    plt.savefig("pictures/part1_accuracy.png")
    plt.show()
    print("show")


if __name__ == '__main__':
    decision_tree(8)
