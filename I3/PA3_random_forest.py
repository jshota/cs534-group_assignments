import matplotlib.pyplot as plt
from collections import Counter
import random
import numpy as np
from scipy.stats import mode


def read(file):
    data = np.genfromtxt(file, dtype=np.str, delimiter=',')
    '''
    for i in range(len(data[0]) - 1):
        if data[0][i] == 'veil-type_p':
            data = np.delete(data, i, axis=1)
    '''
    data = np.delete(data, 0, axis=0)
    data_y = data[:, -1]
    data_x = np.delete(data, -1, axis=1)
    return data_x, data_y


def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


def entropy(Y):
    distribution = Counter(Y)
    s = 0.0
    total = len(Y)
    for y, num_y in distribution.items():
        probability_y = (num_y / total)
        s += (probability_y) * np.log(probability_y)
    return -s


def information_gain(y, y_true, y_false):
    return entropy(y) - (entropy(y_true) * len(y_true) + entropy(y_false) * len(y_false)) / len(y)


class RandomForestClassifier(object):

    def __init__(self, n_estimators=1, max_features=10, max_depth=10,
                 min_samples_split=2, bootstrap=0.4):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.bootstrap = bootstrap
        self.forest = []

    def fit(self, x, y):
        self.forest = []
        n_samples = len(y)
        n_sub_samples = round(n_samples * self.bootstrap)

        for i in range(self.n_estimators):
            shuffle_in_unison(x, y)
            x_subset = x[:n_sub_samples]
            y_subset = y[:n_sub_samples]

            tree = DecisionTreeClassifier(self.max_features, self.max_depth,
                                          self.min_samples_split)
            tree.fit(x_subset, y_subset)
            self.forest.append(tree)


    def predict(self, x):
        n_samples = x.shape[0]
        n_trees = len(self.forest)
        predictions = np.empty([n_trees, n_samples])
        for i in range(n_trees):
            predictions[i] = self.forest[i].predict(x)

        return mode(predictions)[0][0]


    def score(self, x, y):
        y_predict = self.predict(x)
        n_samples = len(y)
        correct = 0
        for i in range(n_samples):
            if str(int(y_predict[i])) == y[i]:
                correct = correct + 1
        accuracy = correct / n_samples
        return accuracy


class DecisionTreeClassifier(object):

    def __init__(self, max_features=10, max_depth=10,
                 min_samples_split=2):

        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split


    def fit(self, x, y):
        n_features = x.shape[1]
        n_sub_features = self.max_features
        feature_indices = random.sample(range(n_features), n_sub_features)
        self.trunk = self.build_tree(x, y, feature_indices, 0)


    def predict(self, X):
        num_samples = X.shape[0]
        y = np.empty(num_samples)
        for j in range(num_samples):
            node = self.trunk

            while isinstance(node, Node):
                if X[j][node.feature_index] != node.threshold:
                    node = node.branch_true
                else:
                    node = node.branch_false
            y[j] = node

        return y


    def build_tree(self, x, y, feature_indices, depth):

        if depth is self.max_depth or len(y) < self.min_samples_split or entropy(y) is 0:
            return mode(y)[0][0]

        feature_index, threshold = find_split(x, y, feature_indices)

        x_true, y_true, x_false, y_false = split(x, y, feature_index, threshold)

        if y_true.shape[0] is 0 or y_false.shape[0] is 0:
            return mode(y)[0][0]

        branch_true = self.build_tree(x_true, y_true, feature_indices, depth + 1)
        branch_false = self.build_tree(x_false, y_false, feature_indices, depth + 1)
        return Node(feature_index, threshold, branch_true, branch_false)



def find_split(x, y, feature_indices):
    #num_features = x.shape[1]

    best_gain = 0
    best_feature_index = 0
    best_threshold = '0'
    for i, feature_index in enumerate(feature_indices):

        values = sorted(set(x[:, feature_index]))
        for j in range(len(values) - 1):
            threshold = '0'
            x_true, y_true, x_false, y_false = split(x, y, feature_index, threshold)
            gain = information_gain(y, y_true, y_false)

            if gain > best_gain:
                best_gain = gain
                best_feature_index = feature_index
                best_threshold = threshold

    return best_feature_index, best_threshold


class Node(object):

    def __init__(self, feature_index, threshold, branch_true, branch_false):
        self.feature_index = feature_index
        self.threshold = threshold
        self.branch_true = branch_true
        self.branch_false = branch_false


def split(x, y, feature_index, threshold):
    x_true = []
    y_true = []
    x_false = []
    y_false = []

    for j in range(len(y)):
        if x[j][feature_index] != threshold:
            x_true.append(x[j])
            y_true.append(y[j])
        else:
            x_false.append(x[j])
            y_false.append(y[j])

    x_true = np.array(x_true)
    y_true = np.array(y_true)
    x_false = np.array(x_false)
    y_false = np.array(y_false)

    return x_true, y_true, x_false, y_false


if __name__ == '__main__':
    for i in range(10):
        x_train, y_train = read('pa3_train.csv')
        x_valid, y_valid = read('pa3_val.csv')

        d = 2
        m = 5
        # ms = [1, 2, 5, 10, 25, 50]

        n = [1, 2, 5, 10, 25]
        # n_trees = 15
        #n=[10]
        train_acc = []
        valid_acc = []
        for n_trees in n:
        # for m in ms:
            print('Training data with the number of tress = ', n_trees)
            forest = RandomForestClassifier(n_estimators=n_trees, max_features=m, max_depth=d,
                                            min_samples_split=2, bootstrap=0.1)
            forest.fit(x_train, y_train)

            train_accuracy = forest.score(x_train, y_train)
            print('The accuracy was', 100 * train_accuracy, '% on the test data.')
            valid_accuracy = forest.score(x_valid, y_valid)
            print('The accuracy was', 100 * valid_accuracy, '% on the test data.')
            train_acc.append(100 * train_accuracy)
            valid_acc.append(100 * valid_accuracy)

            # plot
            # plt.plot(ms, train_acc, 'b', linewidth=1, label="Training_accuracy")
            # plt.plot(ms, valid_acc, 'r', linewidth=1, label="Valid_accuracy")
        plt.plot(n, train_acc, 'b', linewidth=1, label="Training_accuracy")
        plt.plot(n, valid_acc, 'r', linewidth=1, label="Valid_accuracy")
        # plt.xlabel("m")
        plt.xlabel("n")
        plt.ylabel("accuracy")
        plt.legend(loc='lower right')
        plt.title("Random_forest")
        # plt.savefig("pictures/part2_accuracy_ms" + str(i) + ".png")
        plt.savefig("pictures/part2_accuracy_all" + str(i) + ".png")
        plt.show()
