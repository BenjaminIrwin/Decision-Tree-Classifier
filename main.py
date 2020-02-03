import numpy as np


from classification import DecisionTreeClassifier
from eval import Evaluator

def data_split(x, y, k):
    # split the data in k parts
    xpart = x.split(k);
    ypart = y.split(k);

    return xpart, ypart


def cross_validation(x, y, k):
    xpart, ypart = data_split(k)
    accuracy = np.zeros(k)

    for i in range(0, k):

        # split data correctly
        xval = xpart[i]
        yval = ypart[i]
        xtrain = np.delete(xpart, i)
        ytrain = np.delete(ypart, i)

        # train on training slice
        classifier = DecisionTreeClassifier()
        classifier = classifier.train(xpart, ypart)

        # validate using statistics
        eval = Evaluator()
        confusion = eval.confusion_matrix(predictions, y)
        accuracy[i] = eval.accuracy(confusion)

        # store the maximum classifier
        if accuracy[i] == max(accuracy):
            maxClassifier = classifier

    return accuracy, maxClassifier



if __name__ == "__main__":

    filename = "data/toy.txt"
    classifier = DecisionTreeClassifier()
    x,y = classifier.load_data(filename)
    classifier.evaluate_input(x,y)

    print("Training the decision tree...")
    classifier = classifier.train(x,y)

    print("Loading the test set...")
    x_test = np.array([
        [1, 6, 3],
        [0, 5, 5],
        [1, 5, 0],
        [2, 4, 2]
    ])

    y_test = np.array(["A", "A", "C", "C"])

    #filename = "data/test.txt"
    #x_test,y_test = classifier.load_data(filename)
    predictions = classifier.predict(x_test)


    eval = Evaluator()
    confusion = eval.confusion_matrix(predictions, y)

    # Check whether the confusion matrix works
    x_test = x
    predictions = classifier.predict(x_test)
    eval = Evaluator()
    confusion = eval.confusion_matrix(predictions, y)

    # Check whether cross-validation works
    tup = cross_validation(x, y,2)




