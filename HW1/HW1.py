from __future__ import division

import matplotlib.pyplot as plt
from sklearn import svm
from scipy import io
from sklearn.neighbors import KernelDensity
import numpy as np
from sklearn.model_selection import KFold
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors

X_train = []
X_test = []
y_train = []
y_test = []
X_total = []
y_total = []

'''
    Helper functions
'''
def import_data(file):
    # Loading raw data from the matlab object
    data = (io.loadmat(file))

    # Converting to an array
    dataset = data['x']['data'].tolist()[0][0]
    dataset = np.array(dataset)

    # Getting labels, as a list of lists
    labels = data['x']['nlab'].tolist()[0][0]

    # Flattening them out into one list
    labels = [item for sublist in labels for item in sublist]
    labels = np.array(labels) # 183 are 1 (outliers) and 238 are 2 (inliers)

    return dataset, labels


def remove_anomalies(dataset, labels):
    anomaly_indices = np.where(np.array(labels) == 1)[0].tolist()
    return np.delete(dataset, anomaly_indices, 0)


'''
    Kernel Density Estimation using sci-kit learn
'''
def kernel_density():
    # KDE
    kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(X_train)
    score_samples_log = kde.score_samples(X_test)
    score_samples = np.exp(score_samples_log)

    # plt.plot(score_samples)
    # plt.show()
    accuracy = 0

    for i in range(0, len(score_samples)):
        if score_samples[i] == 0.0 and y_test[i] == 1:
            accuracy += 1
        if score_samples[i] != 0.0 and y_test[i] == 2:
            accuracy += 1

    return accuracy/len(score_samples)


'''
    One Class SVM using sci-kit learn
'''
def one_class_svm():
    total = len(y_test)
    correct = 0

    clf = svm.OneClassSVM(nu=0.9)
    clf.fit(X_train)
    prediction = clf.predict(X_test)
    # prediction = [1 if x == -1.0 else x for x in prediction]  # If its -1.0 then its an outlier
    # prediction = [2 if x == 1.0 else x for x in prediction]  # If its 1.0 then its an inlier

    for i in range(0, total):
        if prediction[i] == -1.0 and y_test[i] == 1:
            correct += 1
        if prediction[i] == 1 and y_test[i] == 2:
            correct += 1

    # print "Total Tests: {}".format(total)
    # print "Correct Predictions: {}".format(correct)
    # print "Accuracy: {}%".format(round((correct/total)*100, 2))

    return correct/total


'''
    Implementation of Local Outlier Factor
'''
def local_reachability_distance(nn_indices, distance_data, k):
    rd = 0
    point_index = nn_indices[0]
    neighbour_indices = nn_indices[1:k+1]

    for distance_index in range(0, len(neighbour_indices)):
        true_distance = distance_data[point_index][distance_index + 1]
        k_distance = max(distance_data[point_index])
        rd += max(k_distance, true_distance)
        distance_index += 1

    return 1 / (rd / k)


def local_outlier_factor():
    # find k-nearest neighbours of a point
    lofs = []
    k = 3
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(X_total)
    nn_distances, nn_indices = nbrs.kneighbors(X_total)

    for index in nn_indices:
        point_index = index[0]
        neighbour_indices = index[1:k+1]

        lrd_sum = 0
        for neighbour_index in neighbour_indices:
            lrd_sum += local_reachability_distance(nn_indices[neighbour_index], nn_distances, k)

        normalized_lrd_n = lrd_sum / k
        lrd_point = local_reachability_distance(nn_indices[point_index], nn_distances, k)
        lofs.append(normalized_lrd_n / lrd_point)

    accuracy = 0
    threshold = 1.2
    for i in range(0, len(lofs)):
        if lofs[i] > threshold and y_total[i] == 1:
            accuracy += 1
        if lofs[i] <= threshold and y_total[i] == 2:
            accuracy += 1

    return accuracy/len(lofs)


'''
    Main function. Start reading the code here
'''
def main():
    # These are the global variables that will be used to hold labels and data
    global X_test
    global X_train
    global y_train
    global y_test
    global X_total
    global y_total

    # Number of splits for cross validation
    num_splits = 3

    # Load data from .mat object
    X_total, y_total = import_data('oc_514.mat')

    # Make a kfold object that will split data into k training and test sets
    kfold = KFold(n_splits=num_splits)

    # Define "classifiers" to be used
    classifiers = {
        "Kernel Density Estimation": kernel_density,
        "One Class SVM": one_class_svm}

    for name, classifier in classifiers.items():
        accuracy = 0
        for train_index, test_index in kfold.split(X_total):

            # Use indices to seperate out training and test data
            X_train, X_test = X_total[train_index], X_total[test_index]
            y_train, y_test = y_total[train_index], y_total[test_index]

            # Every classifier returns an accuracy. We sum and average these for each one
            accuracy += classifier()

        total = accuracy / num_splits
        print "Accuracy of {} is {} %".format(name, round((total)*100, 5))

    accuracy = local_outlier_factor()
    print "Accuracy of {} is {} %".format("Local Outlier Factor", round((accuracy) * 100, 5))


if __name__ == "__main__":
    main()


    # estimator = stats.gaussian_kde(dataset)
    #
    # print estimator.evaluate(dataset)