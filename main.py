import numpy as np


from classification import DecisionTreeClassifier
from eval import Evaluator

"""
def data_split(x, y, k):

    # split the data in k parts
    xpart = np.split(x,k)
    ypart = np.split(y,k)
        
    return np.array(xpart), np.array(ypart)


def cross_validation(x, y, k):


    print(x)
    print(y)
    data = np.concatenate((x, y.T), axis=1)
    np.random.shuffle(data)
    print(data.shape)
    xpart, ypart = data_split(x,y,k)
    accuracy = np.zeros(k)

"""

def data_split(x, y, k):

    #concatenate and randomise data
    data = np.array(np.concatenate((x, y), axis=1))
    np.random.shuffle(data)
    data = np.array(np.split(data,k))

    print(data.shape)
    #split and return
    xpart = np.array(data[:,:,:-1],dtype=float)
    ypart = np.array(data[:,:,-1],dtype=str)

    return xpart, ypart


def cross_validation(x, y, k):    
    
    #print(data.shape)
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

    tree = classifier.root_node
    #print(tree)
    left_height = classifier.node_height(tree["left"])
    right_height = classifier.node_height(tree["right"])
   # print(left_height)
    #print(right_height)
    #classifier.print_tree(tree)

    new_tree = classifier.prune_wrapper(tree, "data/validation.txt")
    #classifier.print_tree(new_tree)
    
    #Nick Testing
    #pruning reduces leaves from 274 to 69 --> accuracy from 0.89 to 0.93
    print(classifier.count_leaves(tree))
    print(classifier.count_leaves(new_tree))
    
    # pruning reduces accuracy on test set from 0.865 to 0.795
    filename = "data/test.txt"
    x_test,y_test = classifier.load_data(filename)
    eval = Evaluator()
    predictions_old = classifier.predict(x_test)
    predictions_new = classifier.predict(x_test, True, new_tree)
    confusion_old = eval.confusion_matrix(predictions_old, y_test)
    confusion_new = eval.confusion_matrix(predictions_new, y_test)
    accuracy_old = eval.accuracy(confusion_old)
    accuracy_new = eval.accuracy(confusion_new)
    print(accuracy_old)
    print(accuracy_new)


    # Check whether the confusion matrix works
    #predictions = classifier.predict(x_test)
    #print("xtest: ", x_test)
    #eval = Evaluator()
    #confusion = eval.confusion_matrix(predictions, y_test)
    # Check whether cross-validation works
   
    tup = cross_validation(x,y,2)
    print(tup)
   
