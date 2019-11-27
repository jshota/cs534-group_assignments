import numpy as np
import csv
from matplotlib import pyplot as plt

from random import sample, randint


def fileRead(fileName):
	data = np.genfromtxt(fileName, delimiter = ',')
	return data

def changeData(data):
    for i in range(len(data[0]) - 1):
        if data[0][i] == 'veil-type_p':
            data = np.delete(data, i, axis=1)
    data = np.delete(data, 0, axis=0)
    for i in range(len(data)):
        if data[:, -1][i] == 0:
            data[:, -1][i] = -1
    cls = data[:, -1]
    data = np.delete(data, -1, axis=1)
    data = np.insert(data, 0, cls, axis=1)
    return data

def changeTest(data):
    for i in range(len(data[0]) - 1):
        if data[0][i] == 'veil-type_p':
            data = np.delete(data, i, axis=1)
    data = np.delete(data, 0, axis=0)
    return data


class DecisionTreeClassifier:

	def __init__(self):
		self.leftNode = None
		self.rightNode = None
		self.splitFeature = None
		self.threshold = None
		self.prediction = None
		self.is_leaf = None

	def insert(self, splitFeature, threshold, is_leaf, prediction):
		self.splitFeature = splitFeature
		self.threshold = threshold
		self.is_leaf = is_leaf
		self.prediction = prediction

'''
	def printTree(self, spacing=""):
		if self.is_leaf:
			print(spacing + "Predict", self.prediction)
			return
		print("Feature : ", self.splitFeature, " Threshold : ", self.threshold)
		print(spacing + '--> True:')
		self.left.printTree()
		print(spacing + '--> False:')
		self.right.printTree()
'''


def predictLabel(tree, example, currentDepth, maximumDepth):
	if tree.is_leaf:
		return tree.prediction
	if currentDepth == maximumDepth:
		return tree.prediction

	if example[tree.splitFeature].reshape(1,1) >= tree.threshold:
		return predictLabel(tree.left, example, currentDepth + 1, maximumDepth)
	else:
		return predictLabel(tree.right, example, currentDepth + 1, maximumDepth)



def calcLabel(data):
    size = len(data)

    sortedValues = data[np.argsort(data[:, 0])]
    splitMat = np.split(sortedValues, np.where(sortedValues[:, 0] > 0)[0][:1])
    sumWeights = np.sum(sortedValues[:, 1])

    p_neg = 0
    p_pos = 0
    if len(splitMat[0]) >= 1 and splitMat[0][0, 0] == -1:
        p_neg = (np.sum(splitMat[0][:, 1]) * len(splitMat[0]) * 1.0) / (sumWeights * size)
        if len(splitMat) > 1:
            p_pos = (np.sum(splitMat[1][:, 1]) * len(splitMat[1]) * 1.0) / (sumWeights * size)
    else:
        p_pos = (np.sum(splitMat[1][:, 1]) * len(splitMat[1]) * 1.0) / (sumWeights * size)

    if p_pos > p_neg:
        prediction = 1
    else:
        prediction = -1

    return prediction


def giniIndex(data):
    size = len(data)
    if size == 0:
        return 0
    sortedValues = data[np.argsort(data[:, 0])]
    splitMat = np.split(sortedValues, np.where(sortedValues[:, 0] == 1)[0][:1])
    if len(splitMat[0]) == 0 or len(splitMat) == 1:
        return 0

    sumWeights = np.sum(sortedValues[:, 1])
    p_pos = (np.sum(splitMat[1][:, 1]) * len(splitMat[1]) * 1.0) / (sumWeights * size)
    p_neg = (np.sum(splitMat[0][:, 1]) * len(splitMat[0]) * 1.0) / (sumWeights * size)

    return (1 - p_neg ** 2 - p_pos ** 2)


def getInfoGain(label, data, u_root):
    value = np.empty([len(data), 3])
    value[:, 0] = label[:, 0]
    value[:, 1] = label[:, 1]
    value[:, 2] = data
    sortedValues = value[np.argsort(value[:, 2])]
    gain = 0
    threshold = 0
    prev_label = 0
    sumWeights = np.sum(sortedValues[:, 1])

    for index in range(len(sortedValues)):
        row = sortedValues[index, :]
        thresh = row[2]
        if prev_label != row[0]:
            if index != 0:
                thresh = (sortedValues[index - 1, 2] + thresh) / 2

            val = np.split(sortedValues, np.where(sortedValues[:, 2] >= thresh)[0][:1])
            trueExamples = val[1]
            falseExamples = val[0]

            u_left = giniIndex(trueExamples[:, 0:2])
            u_right = giniIndex(falseExamples[:, 0:2])
            p_left = (np.sum(trueExamples[:, 1]) * len(trueExamples) * 1.0) / (sumWeights * len(sortedValues))
            p_right = (np.sum(falseExamples[:, 1]) * len(falseExamples) * 1.0) / (sumWeights * len(sortedValues))

            currentGain = u_root - p_left * u_left - p_right * u_right

            if currentGain > gain:
                gain = currentGain
                threshold = thresh
            prev_label = row[0]
    return gain, threshold


def createTree_adaboost(data, maximumDepth, currentDepth, tree):
    if currentDepth == maximumDepth:
        label = calcLabel(data[:, 0:2])
        tree.insert(None, None, True, label)
        return

    u_root = giniIndex(data[:, 0:2])

    gain = 0
    threshold = 0
    best_feature = 0
    for featureIndex in range(2, data.shape[1]):
        currentGain, currentThreshold = getInfoGain(data[:, 0:2], data[:, featureIndex], u_root)
        if currentGain > gain:
            gain = currentGain
            threshold = currentThreshold
            best_feature = featureIndex

    if gain == 0:
        label = calcLabel(data[:, 0:2])
        tree.insert(None, None, True, label)
        return

    trueExamples = data[data[:, best_feature] >= threshold]
    falseExamples = data[data[:, best_feature] < threshold]

    label = calcLabel(data[:, 0:2])
    tree.insert(best_feature, threshold, False, label)
    tree.left = DecisionTreeClassifier()
    tree.right = DecisionTreeClassifier()
    currentDepth = currentDepth + 1
    createTree_adaboost(trueExamples, maximumDepth, currentDepth, tree.left)
    createTree_adaboost(falseExamples, maximumDepth, currentDepth, tree.right)


def errorCalc(tree, data, maxDepth):
    error = 0
    signed_product_y_hypo = []
    for row in data:
        prediction = predictLabel(tree, row, 0, maxDepth)

        if row[0] != prediction:
            error = error + row[1]
            hypothesis_times_y = 1
        else:
            hypothesis_times_y = -1
        signed_product_y_hypo.append(hypothesis_times_y)
    error = error / np.sum(data[:, 1])
    return error, signed_product_y_hypo


def adaboost(data, l, maximumDepth):
    size = len(data)
    D = np.empty(size)
    D.fill(1.0 / size)

    data = np.insert(data, 1, D, axis=1)

    tree_list = []
    alpha_list = []

    for weakLearner in range(l):
        tree = DecisionTreeClassifier()
        createTree_adaboost(data, maximumDepth, 0, tree)
        err, weightChange_list = errorCalc(tree, data, maximumDepth)
        alpha = (np.log(((1 - err) * 1.0) / err)) / 2
        data[:, 1] = data[:, 1] * np.exp(alpha * np.array(weightChange_list))
        tree_list.append(tree)
        alpha_list.append(alpha)

    return tree_list, alpha_list


def treeAccuracy_ada(tree_list, alpha_list, data, maxDepth):
    size = len(data)
    error = 0
    D = np.empty(size)
    data = np.insert(data, 1, D, axis=1)

    for row in data:
        sumWeights = 0
        index = 0
        for tree in tree_list:
            alpha = alpha_list[index]
            prediction = predictLabel(tree, row, 0, maxDepth)
            sumWeights = sumWeights + prediction * alpha
            index = index + 1
        if np.sign(sumWeights) != row[0]:
            error = error + 1

    accuracy = ((size - error) * 1.0 / size) * 100
    return accuracy


def predict(tree_list, alpha_list, data, maxDepth):
    size = len(data)
    error = 0
    D = np.empty(size)
    data = np.insert(data, 0, D, axis=1)
    data = np.insert(data, 0, 0, axis=1)

    for row in data:
        sumWeights = 0
        index = 0
        for tree in tree_list:
            alpha = alpha_list[index]
            prediction = predictLabel(tree, row, 0, maxDepth)
            sumWeights = sumWeights + prediction * alpha
            index = index + 1
        row[0] = np.sign(sumWeights)

    np.savetxt('predictbest.txt', data[:, 0])


if __name__ == '__main__':
    trainData = fileRead('pa3_train.csv')
    trainData = changeData(trainData)
    validData = fileRead('pa3_val.csv')
    validData = changeData(validData)
    testData = fileRead('pa3_test.csv')
    testData = changeTest(testData)
    train_acc_list = []
    valid_acc_list = []
    max_depth = 2

    print("!!!!!Executing ADABOOST!!!!!")
    #itr_list = [1, 2, 5, 10, 15]
    itr_list = [6]
    i = 0
    for l in itr_list:
	    print("``````````````````````For {} Weak-Learners```````````````````````````".format(l))
	    #itr_list.append(l)
	    tree_list, alpha_list = adaboost(trainData, l, max_depth)
	    train_acc_list.append(treeAccuracy_ada(tree_list, alpha_list, trainData, max_depth))
	    valid_acc_list.append(treeAccuracy_ada(tree_list, alpha_list, validData, max_depth))
	    i = i + 1
    predict(tree_list, alpha_list, testData, max_depth)
    # np.savetxt('predictbest.txt', tree_list)
    # plt.scatter(itr_list, train_acc_list, color = 'blue', s = 5)
    # blue_line, = plt.plot(itr_list, train_acc_list, color = 'blue', label = 'Training Accuracy')
    # plt.title("AdaBoost Accuracy")
    # plt.xlabel("L")
    # plt.ylabel("Accuracy")
    # #
    # plt.scatter(itr_list, valid_acc_list, color = 'red', s = 5)
    # red_line, = plt.plot(itr_list, valid_acc_list, color = 'red', label = 'Validation Accuracy')
    # plt.legend(handles  = [blue_line, red_line])
    # plt.grid()
    # plt.savefig("./part3.png")
    # plt.show()
    # print(train_acc_list)
    # print(valid_acc_list)
