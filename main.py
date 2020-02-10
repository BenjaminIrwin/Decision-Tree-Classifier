import numpy as np


from classification import DecisionTreeClassifier
from eval import Evaluator
from pruning import Pruning
import copy

def data_split(x, y, k):

    data = np.array(np.concatenate((x, y), axis=1))
    np.random.shuffle(data)
    data = np.array(np.split(data,k))

    #split and return
    xpart = np.array(data[:,:,:-1],dtype=int)
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
        xtrain = np.delete(xpart,i,0).reshape((k-1)*xval.shape[0],xval.shape[1])
        ytrain = np.delete(ypart,i,0).reshape((k-1)*xval.shape[0],1)


        # train on training slice
        classifiers[i] = DecisionTreeClassifier()
        classifiers[i] = classifiers[i].train(xtrain, ytrain)

        #predict for test class
        predictions = classifiers[i].predict(xval)

        # validate using statistics
        eval = Evaluator()
        confusion = eval.confusion_matrix(predictions, yval)
        accuracy[i] = eval.accuracy(confusion)

    return accuracy, classifiers

def weighted_predict(classifiers,x_test):

    predictions = np.zeros([len(classifiers),len(x_test)],dtype=object)
    result = np.zeros(len(x_test),dtype=object)

    for i in range(len(classifiers)):
        predictions[i,:] = classifiers[i].predict(x_test)


    for i in range(predictions.shape[1]):
        vals, counts = np.unique(predictions[:,i], return_counts = True)
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
    
    #QUESTION 1
    print("Question 1")
    print("Loading the data")
    
    filename = "data/train_full.txt"
    classifier = DecisionTreeClassifier()
    x,y = classifier.load_data(filename)
    

    #QUESTION 2
    print("Question 2")
    print("Training the tree with two different methods")

    print("Training the decision tree...")
    classifier = classifier.train(x,y)
    
    print("Loading the test set...")

    filename = "data/test.txt"
    x_test, y_test = classifier.load_data(filename)
    
    print("\nPredicting on test.txt data with 4 different trees")
    
    #Load the evaulator class
    eval = Evaluator()
    prune = Pruning()

    print("\nTree 1 Unpruned")
    tree = np.load('initial_tree.npy',allow_pickle = True).item()
    predictions = classifier.predict(x_test,tree)
    confusion = eval.confusion_matrix(predictions, y_test)
    accuracy_1 = eval.accuracy(confusion)
    print("number of leaves:" ,prune.count_leaves(tree))
    print("Tree 1 Unpruned Accuracy: " + str(np.round(accuracy_1*100,2)))
    
    print("\nTree 1 pruned")
    tree_2 = np.load('initial_tree_pruned.npy',allow_pickle = True).item()
    predictions = classifier.predict(x_test,tree_2)
    confusion = eval.confusion_matrix(predictions, y_test)
    accuracy_2 = eval.accuracy(confusion)
    print("number of leaves:" ,prune.count_leaves(tree_2))
    print("Tree 1 pruned Accuracy: " + str(np.round(accuracy_2*100,2)))
    
    print("\nTree 2 unpruned")
    tree_3 = np.load('simple_tree.npy',allow_pickle = True).item()
    predictions = classifier.predict(x_test)
    confusion = eval.confusion_matrix(predictions, y_test)
    accuracy_3 = eval.accuracy(confusion)
    print("number of leaves:",prune.count_leaves(tree_3))
    print("Tree 2 unpruned Accuracy: " + str(np.round(accuracy_3*100,2)))
    
    print("\nTree 2 pruned")
    tree_4 = np.load('simple_tree_pruned.npy',allow_pickle = True).item()
    predictions = classifier.predict(x_test,tree_4)
    confusion = eval.confusion_matrix(predictions, y_test)
    accuracy_4 = eval.accuracy(confusion)
    print("number of leaves:",prune.count_leaves(tree_4))
    print("Tree 2 pruned Accuracy: " + str(np.round(accuracy_4*100,2)))
    
    print("Question 2.3")
    print("Printing the tree")
    classifier.print_tree(tree_2,"Method_1_Pruned")
    classifier.print_tree(tree_3,"Method_2_UnPruned")
    
    print("\n\n")
    
    #### QUESTION 3 ##########
    print("Question 3")
    filename = "data/test.txt"
    classifier = DecisionTreeClassifier()
    x_test,y_test = classifier.load_data(filename)

    
    #Question 3.1
    print("\nQ3.1")

    filenames = ["data/train_full.txt", "data/train_sub.txt","data/train_noisy.txt"]
    for f in filenames:

        print("\ntraining " + f)
        classifier = DecisionTreeClassifier()
        x,y = classifier.load_data(f)
        classifier = classifier.train(x,y)
        predictions = classifier.predict(x_test)
        print_stats(predictions,y_test)

    #Question 3.3
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

    #Question 3.5
    print("\nQ3.5")
    predictions = weighted_predict(crossval[1],x_test)
    print_stats(predictions,y_test)
    
    #QUESTION 4 - PRUNING
    
    print("QUESTION 4")
    eval = Evaluator()
    print("Method 1: Reduced Error Pruning\n")
    
    test_filename = "data/test.txt"
    val_filename = "data/validation.txt"
    x_val, y_val = classifier.load_data(val_filename)
    x_test,y_test = classifier.load_data(test_filename)
    
    ## METHOD 2's UNPREPRUNED TREE 
    print("Pruning Method 2's UnPruned Tree")
    reduced_error_tree_1 = copy.deepcopy(tree_3)
    new_tree = prune.prune_tree_reduced_error(reduced_error_tree_1, x_val, y_val)
   
    print("number of leaves before:"+ str(prune.count_leaves(reduced_error_tree_1)))
    print("number of leaves after:"+str(prune.count_leaves(new_tree)))
    
    
    predictions_old = classifier.predict(x_test)
    predictions_new = classifier.predict(x_test,new_tree)
    confusion_old = eval.confusion_matrix(predictions_old, y_test)
    confusion_new = eval.confusion_matrix(predictions_new, y_test)
    accuracy_old = eval.accuracy(confusion_old)
    accuracy_new = eval.accuracy(confusion_new)
    print("\nOld accuracy on test set:" + str(accuracy_3))
    print("New accuracy on test set:" + str(accuracy_new))
    
    ## METHOD 1's PREPRUNED TREE 
    
    print("\nPruning the Method 1's Prepruned tree\n")
    reduced_error_tree_2 = copy.deepcopy(tree_2)
    new_tree = prune.prune_tree_reduced_error(reduced_error_tree_2, x_val, y_val)
    print("\nnumber of leaves before:"+ str(prune.count_leaves(reduced_error_tree_2)))
    print("number of leaves after:"+str(prune.count_leaves(new_tree)))
   
    predictions_old = classifier.predict(x_test)
    predictions_new = classifier.predict(x_test,new_tree)
    confusion_old = eval.confusion_matrix(predictions_old, y_test)
    confusion_new = eval.confusion_matrix(predictions_new, y_test)
    accuracy_old = eval.accuracy(confusion_old)
    accuracy_new = eval.accuracy(confusion_new)
    print("\nOld accuracy on test set:" + str(accuracy_2))
    print("New accuracy on test set:" + str(accuracy_new))
    
    
    ##POST CHI^2 PRUNING METHOD#####
    
    print("\n")
    print("\nMethod 2: Post CHI^2 Pruning\n")
    ## METHOD 2's UNPREPRUNED TREE 
    print("\nPruning Method 2's UnPruned Tree") 
    chi_1_tree = prune.post_chi_pruning(tree_3)
    predictions_new = classifier.predict(x_test,chi_1_tree)
    confusion_new = eval.confusion_matrix(predictions_new, y_test)
    accuracy_new = eval.accuracy(confusion_new)
    print("\nOld accuracy on test set:" + str(accuracy_3))
    print("New accuracy on test set:" + str(accuracy_new))
    
    ## METHOD 1's PREPRUNED TREE 
    print("\nPruning Method 1's Pre Pruned Tree") 
    chi_2_tree = prune.post_chi_pruning(tree_2)
    predictions_new = classifier.predict(x_test,chi_2_tree)
    confusion_new = eval.confusion_matrix(predictions_new, y_test)
    accuracy_new = eval.accuracy(confusion_new)
    print("\nOld accuracy on test set:" + str(accuracy_2))
    print("New accuracy on test set:" + str(accuracy_new))
    
    
    
    
    
    
    

    
    




