# team 76
# Panagiotis Boulotis   AM: 4271
# Argyrios Zezos        AM: 4588

import pandas as pd  # for csv reading
import numpy as np  # for csv writing
import statistics as st  # for mean() and stdev()
from math import exp, sqrt, pi  # for the normal pdf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm  # Import svm model
from sklearn.model_selection import GridSearchCV


train = pd.read_csv(r'train.csv')
# train data handling, splitting the columns to different lists
train_data = pd.DataFrame(train)
x_train = train_data.iloc[:, 1:6].values  # we don't want the id and type values
y_train = train_data['type'].values.tolist()
train_length = len(y_train)  # 371 rows
# for Naive Bayes
train_stats = list()
prob_class = list()
sorted_x_train = list()
color_prob = [[0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0]]

# convert the color column to numbers
for i in range(len(x_train)):
    if x_train[i][4] == 'clear':
        x_train[i][4] = 1 / 6
    elif x_train[i][4] == 'black':
        x_train[i][4] = 2 / 6
    elif x_train[i][4] == 'green':
        x_train[i][4] = 0.5
    elif x_train[i][4] == 'white':
        x_train[i][4] = 4 / 6
    elif x_train[i][4] == 'blood':
        x_train[i][4] = 5 / 6
    elif x_train[i][4] == 'blue':
        x_train[i][4] = 1

# same for test.csv
test = pd.read_csv(r'test.csv')
test_data = pd.DataFrame(test)
test_id = test_data['id'].values.tolist()
x_test = test_data.iloc[:, 1:6].values
test_length = len(test_id)  # 529 rows

# convert the color column to numbers
for i in range(len(x_test)):
    if x_test[i][4] == 'clear':
        x_test[i][4] = 1 / 6
    elif x_test[i][4] == 'black':
        x_test[i][4] = 2 / 6
    elif x_test[i][4] == 'green':
        x_test[i][4] = 0.5
    elif x_test[i][4] == 'white':
        x_test[i][4] = 4 / 6
    elif x_test[i][4] == 'blood':
        x_test[i][4] = 5 / 6
    elif x_test[i][4] == 'blue':
        x_test[i][4] = 1


def knn(k):
    # KNearestNeighbours with the use of the euclidean distance
    classifier = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    classifier.fit(x_train, y_train)  # the classifier uses the train data needed
    y_pred = classifier.predict(x_test)  # y_prediction is the missing type of the test samples

    # for the Kaggle accuracy
    knn_results = [['id', 'type']]  # the required labels
    for j in range(len(y_pred)):
        knn_results.append([test_id[j], y_pred[j]])
    return knn_results


def knn_execute():
    results1 = knn(1)  # k=1
    np.savetxt("knn1.csv", results1, delimiter=",", fmt='% s')

    results2 = knn(3)  # k=3
    np.savetxt("knn3.csv", results2, delimiter=",", fmt='% s')

    results3 = knn(5)  # k=5
    np.savetxt("knn5.csv", results3, delimiter=",", fmt='% s')

    results4 = knn(10)  # k=10
    np.savetxt("knn10.csv", results4, delimiter=",", fmt='% s')


def mlp1(k, epochs):
    global x_train, y_train, x_test
    # We need to create the mlp network
    network = Sequential()
    # the input variables are 5, the number of columns
    # the hidden layer has K neurons uses sigmoid activation function
    network.add(Dense(k, input_dim=5, activation='sigmoid'))
    network.add(Dense(3, activation='softmax'))  # the output layer has 3 neurons, as many as the classes
    network.compile(optimizer='sgd', loss='sparse_categorical_crossentropy')

    network.fit(x=x_train, y=y_train, epochs=epochs)
    temp_y_pred = network.predict(x_test)
    y_pred = []
    for j in range(len(temp_y_pred)):
        if temp_y_pred[j][0] == max(temp_y_pred[j]):
            y_pred.append('Ghost')
        elif temp_y_pred[j][1] == max(temp_y_pred[j]):
            y_pred.append('Ghoul')
        elif temp_y_pred[j][2] == max(temp_y_pred[j]):
            y_pred.append('Goblin')

    # for the Kaggle accuracy
    mlp1_results = [['id', 'type']]
    for j in range(len(y_pred)):
        mlp1_results.append([test_id[j], y_pred[j]])
    return mlp1_results


def mlp2(k1, k2, epochs):
    global x_train, y_train, x_test
    # We need to create the mlp network
    network = Sequential()
    network.add(Dense(k1, input_dim=5, activation='sigmoid'))  # the 1st hidden layer with K1 neurons
    network.add(Dense(k2, activation='sigmoid'))  # the 2nd hidden layer with K2 neurons
    network.add(Dense(3, activation='softmax'))
    network.compile(optimizer='sgd', loss='sparse_categorical_crossentropy')

    network.fit(x=x_train, y=y_train, epochs=epochs)
    temp_y_pred = network.predict(x_test)
    y_pred = []
    for j in range(len(temp_y_pred)):
        if temp_y_pred[j][0] == max(temp_y_pred[j]):
            y_pred.append('Ghost')
        elif temp_y_pred[j][1] == max(temp_y_pred[j]):
            y_pred.append('Ghoul')
        elif temp_y_pred[j][2] == max(temp_y_pred[j]):
            y_pred.append('Goblin')

    # for the Kaggle accuracy
    mlp2_results = [['id', 'type']]
    for j in range(len(y_pred)):
        mlp2_results.append([test_id[j], y_pred[j]])
    return mlp2_results


def neural_execute():
    # MLP with 1 Hidden Layer
    results1 = mlp1(50, 350)  # k=50
    np.savetxt("mlp1_50.csv", results1, delimiter=",", fmt='% s')
    results2 = mlp1(100, 350)  # k=100
    np.savetxt("mlp1_100.csv", results2, delimiter=",", fmt='% s')
    results3 = mlp1(200, 400)  # k=200
    np.savetxt("mlp1_200.csv", results3, delimiter=",", fmt='% s')

    # MLP with 2 Hidden Layers
    results = mlp2(50, 25, 700)  # k1=50,k2=25
    np.savetxt("mlp2_50,25.csv", results, delimiter=",", fmt='% s')
    results = mlp2(100, 50, 700)  # k1=100,k2=50
    np.savetxt("mlp2_100,50.csv", results, delimiter=",", fmt='% s')
    results = mlp2(200, 100, 700)  # k1=200,k2=100
    np.savetxt("mlp2_200,100.csv", results, delimiter=",", fmt='% s')


def svm_execute(kernel):
    # Create a svm Classifier and hyper parameter tuning
    ml = svm.SVC()

    # defining parameter range
    param_grid = {'C': [1, 10, 100, 1000, 10000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': [kernel]}  # ['linear', 'rbf']}  linear or gaussian

    grid = GridSearchCV(ml, param_grid, refit=True, verbose=1, cv=15)

    # fitting the model for grid search
    grid_search = grid.fit(x_train, y_train)

    print(grid_search.best_params_)

    accuracy = grid_search.best_score_ * 100
    print("Accuracy for our training dataset with tuning is : {:.2f}%".format(accuracy))

    y_pred = grid.predict(x_test)

    results = [['id', 'type']]
    for j in range(len(y_pred)):
        results.append([test_id[j], y_pred[j]])

    return results


def normal_pdf(x, mean, stdev):
    exponent = exp(-((x-mean)**2 / (2 * stdev**2)))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent


def calculate_color_prob(x_value, class_value):
    if x_value == 1/6:
        return color_prob[class_value][0]
    elif x_value == 2/6:
        return color_prob[class_value][1]
    elif x_value == 0.5:
        return color_prob[class_value][2]
    elif x_value == 4/6:
        return color_prob[class_value][3]
    elif x_value == 5/6:
        return color_prob[class_value][4]
    else:
        return color_prob[class_value][5]


# Calculates the probabilities for the Naive Bayes method
def calculate_prob(row):
    global train_stats, prob_class
    probabilities = [0, 0, 0]
    for class_value in range(3):
        probabilities[class_value] = float(prob_class[class_value])  # class probability
        for column in range(len(row)):
            # measurements needed for normal_pdf
            x_value = row[column]
            mean = train_stats[class_value][column][0]
            stdev = train_stats[class_value][column][1]

            if column != 4:
                # using normal distribution
                probabilities[class_value] *= normal_pdf(x_value, mean, stdev)
            else:
                probabilities[class_value] *= calculate_color_prob(x_value, class_value)

    return probabilities


# The winning class is predicted as the missing class
def nb_predict(row):
    probabilities = calculate_prob(row)
    best_prob = max(probabilities)
    if probabilities[0] == best_prob:
        return 'Ghost'
    elif probabilities[1] == best_prob:
        return 'Ghoul'
    else:
        return 'Goblin'


def naive_bayes():
    global train_stats, sorted_x_train
    # a) sort the x_train list depending on the class
    sorted0 = []
    sorted1 = []
    sorted2 = []
    for n in range(len(x_train)):
        if y_train[n] == 0:
            sorted0.append(x_train[n])
        elif y_train[n] == 1:
            sorted1.append(x_train[n])
        else:
            sorted2.append(x_train[n])

    sorted_x_train = sorted0 + sorted1 + sorted2

    # b) save the stats for each class
    train_stats.append([(st.mean(column), st.stdev(column)) for column in zip(*sorted0)])
    train_stats.append([(st.mean(column), st.stdev(column)) for column in zip(*sorted1)])
    train_stats.append([(st.mean(column), st.stdev(column)) for column in zip(*sorted2)])

    # c) calculating probabilities
    prob_class.append(len(sorted0)/train_length)  # probability a row being in class 0
    prob_class.append(len(sorted1)/train_length)  # same for class 1
    prob_class.append(len(sorted2)/train_length)  # same for class 2

    # color probabilities for each class
    # for class 0
    for j in range(len(sorted0)):
        if sorted0[j][4] == 1/6:
            color_prob[0][0] += (1/len(sorted0))
        elif sorted0[j][4] == 2/6:
            color_prob[0][1] += (1/len(sorted0))
        elif sorted0[j][4] == 0.5:
            color_prob[0][2] += (1/len(sorted0))
        elif sorted0[j][4] == 4/6:
            color_prob[0][3] += (1/len(sorted0))
        elif sorted0[j][4] == 5/6:
            color_prob[0][4] += (1/len(sorted0))
        else:
            color_prob[0][5] += (1/len(sorted0))

    # for class 1
    for j in range(len(sorted1)):
        if sorted1[j][4] == 1 / 6:
            color_prob[1][0] += (1 / len(sorted1))
        elif sorted1[j][4] == 2 / 6:
            color_prob[1][1] += (1 / len(sorted1))
        elif sorted1[j][4] == 0.5:
            color_prob[1][2] += (1 / len(sorted1))
        elif sorted1[j][4] == 4 / 6:
            color_prob[1][3] += (1 / len(sorted1))
        elif sorted1[j][4] == 5 / 6:
            color_prob[1][4] += (1 / len(sorted1))
        else:
            color_prob[1][5] += (1 / len(sorted1))

    # for class 2
    for j in range(len(sorted2)):
        if sorted2[j][4] == 1 / 6:
            color_prob[2][0] += (1 / len(sorted2))
        elif sorted2[j][4] == 2 / 6:
            color_prob[2][1] += (1 / len(sorted2))
        elif sorted2[j][4] == 0.5:
            color_prob[2][2] += (1 / len(sorted2))
        elif sorted2[j][4] == 4 / 6:
            color_prob[2][3] += (1 / len(sorted2))
        elif sorted2[j][4] == 5 / 6:
            color_prob[2][4] += (1 / len(sorted2))
        else:
            color_prob[2][5] += (1 / len(sorted2))

    # predictions using NB
    y_pred = []
    for row in x_test:
        y_pred.append(nb_predict(row))

    results = [['id', 'type']]
    for j in range(len(y_pred)):
        results.append([test_id[j], y_pred[j]])

    np.savetxt("naive_bayes.csv", results, delimiter=",", fmt='% s')


if __name__ == '__main__':
    # K Nearest Neighbour
    # knn_execute()

    # Support Vector Machines
    results_linear = svm_execute('linear')
    np.savetxt("svm_linear.csv", results_linear, delimiter=",", fmt='% s')

    results_rbf = svm_execute('rbf')
    np.savetxt("svm_rbf.csv", results_rbf, delimiter=",", fmt='% s')

    # for compatibility reasons assign an int value to the string type classes
    for j in range(len(y_train)):
        if y_train[j] == 'Ghost':
            y_train[j] = 0
        elif y_train[j] == 'Ghoul':
            y_train[j] = 1
        else:
            y_train[j] = 2

    x_train = x_train.tolist()
    x_test = x_test.tolist()

    # Neural Networks
    # neural_execute()

    # Naive Bayes
    # naive_bayes()
