import numpy as np


from classification import DecisionTreeClassifier
from eval import Evaluator

def data_split(x, y, k):

    #concatenate and randomise data
    data = np.array(np.concatenate((x, y), axis=1))
    np.random.shuffle(data)
    data = np.array(np.split(data,k))

    #split and return
    xpart = np.array(data[:,:,:-1],dtype=float)
    ypart = np.array(data[:,:,-1],dtype=str)

    return xpart, ypart


def cross_validation(x, y, k):    
    

    xpart, ypart = data_split(x,y,k)
    accuracy = np.zeros(k)


    for i in range(k):
        
        # split data correctly
        xval = xpart[i]
        yval = ypart[i]
        xtrain = np.delete(xpart,i,0)[0]
        ytrain = np.delete(ypart,i,0)[0]

        # train on training slice
        classifier = DecisionTreeClassifier()
        classifier = classifier.train(xtrain, ytrain)

        #predict for test class
        predictions = classifier.predict(xval)

        # validate using statistics
        eval = Evaluator()
        confusion = eval.confusion_matrix(predictions, yval)
        accuracy[i] = eval.accuracy(confusion)

        # store the maximum classifier
        if accuracy[i] == max(accuracy):
            maxClassifier = classifier

    return accuracy, maxClassifier


if __name__ == "__main__":

    filename = "data/train_full.txt"
    classifier = DecisionTreeClassifier()
    x,y = classifier.load_data(filename)

    print("Training the decision tree...")
    classifier = classifier.train(x,y)

    print("Loading the test set...")

    #tree = classifier.root_node
    x_test = np.array([
        [1, 6, 3],
        [0, 5, 5],
        [1, 5, 0],
        [2, 4, 2]
    ])

    y_test = np.array(["A", "A", "C", "C"])
    #predictions = classifier.predict(x_test)
    #print(tree)
    filename = "data/test.txt"
    x_test, y_test = classifier.load_data(filename)
    predictions = classifier.predict(x_test)
    
    tree = np.load('tree.npy',allow_pickle = True).item()
    tree_list = classifier.cost_complexity_pruning(tree)

    print(len(tree_list))






