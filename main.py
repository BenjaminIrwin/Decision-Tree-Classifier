import numpy as np


from classification import DecisionTreeClassifier
from eval import Evaluator

def data_split(x, y, k):

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
    classifiers = np.empty(k,dtype=object)

    for i in range(k):

        # split data correctly
        xval = xpart[i]
        yval = ypart[i]
        xtrain = np.delete(xpart, i)
        ytrain = np.delete(ypart, i)

        # train on training slice
        classifiers[i] = DecisionTreeClassifier()
        classifiers[i] = classifiers[i].train(xtrain, ytrain)

        #predict for test class
        predictions = classifiers[i].predict(xval)

        # validate using statistics
        eval = Evaluator()
        confusion = eval.confusion_matrix(predictions, y)
        accuracy[i] = eval.accuracy(confusion)

    return accuracy, classifiers

def weighted_predict(classifiers,x_test):

    predictions = np.zeros([len(x_test),len(classifiers)],dtype=object)
    result = np.zeros(len(x_test),dtype=object)

    for i in range(len(classifiers)):
        predictions[:,i] = classifiers[i].predict(x_test)


    for i in range(predictions.shape[0]):
        vals, counts = np.unique(predictions[i,:], return_counts = True)
        #print("vals counts")
        #print(vals,counts)
        result[i] = vals[np.argmax(counts)]        

    return result


def print_stats(predictions,y_test):
    
    eval = Evaluator()
    confusion = eval.confusion_matrix(predictions, y_test)

    accuracy = eval.accuracy(confusion)
    precision = eval.precision(confusion)
    recall = eval.recall(confusion)
    f1 = eval.f1_score(confusion)

    print("confusion", confusion)
    print("accuracy", accuracy)
    print("precision", precision)
    print("recall", recall)
    print("f1", f1)
 
    
    return


if __name__ == "__main__":

    filename = "data/train_full.txt"
    classifier = DecisionTreeClassifier()
    x,y = classifier.load_data(filename)
    #classifier.evaluate_input(x,y)

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

    filename = "data/test.txt"
    x_test, y_test = classifier.load_data(filename)
    predictions = classifier.predict(x_test)
    
    tree = np.load('tree.npy',allow_pickle = True).item()
    trees = classifier.cost_complexity_pruning(tree)
    
    #print(trees)
    
    #tree,accuracy = classifier.calculate_best_pruned_tree(trees,x_test,y_test)
    #print(tree)
    #print(accuracy)
    #classifier.print_tree(tree)
    
    
    #### QUESTION 3 ##########
    print("Loading the test set...")
    filename = "data/test.txt"
    classifier = DecisionTreeClassifier()
    x_test,y_test = classifier.load_data(filename)

    
    # Question 3.1
    print("\nQ3.1")

    filenames = ["data/train_full.txt", "data/train_sub.txt","data/train_noisy.txt"]
    for f in filenames:

        print("\ntraining " + f)
        classifier = DecisionTreeClassifier()
        x,y = classifier.load_data(f)
        classifier = classifier.train(x,y)
        predictions = classifier.predict(x_test)
        print_stats(predictions,y_test)
    
    
    # Question 3.3
    print("\nQ3.3")
    filename = "data/train_full.txt"
    x,y = classifier.load_data(filename)

    crossval = cross_validation(x,y,10)
    accuracy = crossval[0]
    average_acc = sum(accuracy)/len(accuracy)
    std = np.std(accuracy)

    print("average acc", average_acc)
    print("std", std)
    
    #Question 3.4
    print("\nQ3.4")
    predictions = crossval[1][np.argmax(crossval[0])].predict(x_test)
    print_stats(predictions,y_test)
    q = np.empty([len(predictions),2],dtype=object)
    
    q[:,0] = predictions

    #Question 3.5
    print("\nQ3.5")
    predictions = weighted_predict(crossval[1],x_test)
    print_stats(predictions,y_test)
    q[:,1] = predictions
    #print(q)

    
    
    #Nicks question 4 stuff to make it work
    
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
    predictions_new = classifier.predict(x_test,new_tree)
    confusion_old = eval.confusion_matrix(predictions_old, y_test)
    confusion_new = eval.confusion_matrix(predictions_new, y_test)
    accuracy_old = eval.accuracy(confusion_old)
    accuracy_new = eval.accuracy(confusion_new)
    print(accuracy_old)
    print(accuracy_new)

    
    




