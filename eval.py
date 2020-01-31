##############################################################################
# CO395: Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: 
# Complete the following methods of Evaluator:
# - confusion_matrix()
# - accuracy()
# - precision()
# - recall()
# - f1_score()
##############################################################################

import numpy as np


class Evaluator(object):
    """ Class to perform evaluation
    """
    ## WHAT DATA SHOULD IT BE ABLE TO HANDLE?
    def confusion_matrix(self, prediction, annotation, class_labels=None):
        """ Computes the confusion matrix.
        
        Parameters
        ----------
        prediction : np.array
            an N dimensional numpy array containing the predicted
            class labels
        annotation : np.array
            an N dimensional numpy array containing the ground truth
            class labels
        class_labels : np.array
            a C dimensional numpy array containing the ordered set of class
            labels. If not provided, defaults to all unique values in
            annotation.
        
        Returns
        -------
        np.array
            a C by C matrix, where C is the number of classes.
            Classes should be ordered by class_labels.
            Rows are ground truth per class, columns are predictions.
        """
        
        if not class_labels:
            class_labels = np.unique(annotation)

        confusion = np.zeros((len(class_labels), len(class_labels)), dtype=np.int)

        print("printing prediction")
        print(prediction)
        print("printing annotation")
        print(annotation)

        print("printing empty confusion")
        print(confusion)
        
        for i in range(0,len(annotation)):

            row = np.where(class_labels == annotation[i,0])
            column = np.where(class_labels == prediction[i])

            if(row[0].size!= 0 and column[0].size != 0):  
                confusion[row[0], column[0]] += 1
        
        
        return confusion

    
    def accuracy(self, confusion):
        """
         Computes the accuracy given a confusion matrix.
        
        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions
        
        Returns
        -------
        float
            The accuracy (between 0.0 to 1.0 inclusive)
        """

        # feel free to remove this
        true_total = 0.0
        num_predictions = self.count_predictions(confusion)

        for i in range(confusion.shape[0])
            true_total += confusion[i, i]

         accuracy = true_total/num_predictions
        
        return accuracy

    def count_predictions(self, confusion):

        total = 0
        for i in range(confusion.shape[0]):
            for j in range(confusion.shape[1]):
                total += confusion[i, j]

        return total

    def macro_average(self, evaluate):
        total = 0;
        for i in range(len(evaluate)):
            total += evaluate[i]

        macro_av = total/len(evaluate)

        return macro_av

    def precision(self, confusion):
        
        # Initialise array to store precision for C classes
        p = np.zeros((len(confusion), ))

        true_positive_total = 0;
        for i in range(confusion.shape[0]): #iterate thru each TruePositive (i.e. diagonally starting at confusion[0, 0]
            true_positive_total = confusion[i, i]

            false_positive_total = 0
            for j in range(confusion.shape[0]): #After finding the TruePositive add up all the other values on its column
                if j != i:
                    false_positive_total += confusion[j, i]


            precision = true_positive_total/true_positive_total + false_positive_total
            p[i] = precision

        macro_p = self.macro_average(p)

        return (p, macro_p)
    
    
    def recall(self, confusion):

        # Initialise array to store recall for C classes
        r = np.zeros((len(confusion), ))


        for i in range(len(confusion)):

            total_true_positive = confusion[i, i]
            total_false_negative = 0
            for j in range(len(confusion)):
                for k in range(len(confusion)):
                    if j != k and k != i:
                        total_false_negative += confusion[j, k]

            recall = total_true_positive/(total_true_positive + total_false_negative)
            r[i] = recall


        # You will also need to change this        
        macro_r = self.macro_average(r)
        
        return r, macro_r
    
    
    def f1_score(self, confusion):

        # Initialise array to store recall for C classes
        f = np.zeros((len(confusion), ))
        r = self.recall(confusion)[0]
        p = self.precision(confusion)[0]


        for i in range(len(confusion)):
            f1 = 2*((p[i]*r[i])/(p[i]+r[i]))
            f[i] = f1

        macro_f = self.macro_average(f)
        
        return (f, macro_f)
   
 
