import numpy as np


from classification import DecisionTreeClassifier


if __name__ == "__main__":

    filename = "data/train_full.txt"
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
    #predictions = classifier.predict(x_test)



    #x_test = x
    #predictions = classifier.predict(x)
    #predictions = classifier.predict(x_test)

    #eval = Evaluator()
    #confusion = eval.confusion_matrix(predictions, y)
