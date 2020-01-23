##############################################################################
# CO395: Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the train() and predict() methods of the 
# DecisionTreeClassifier 
##############################################################################

import numpy as np


class DecisionTreeClassifier(object):
    """
    A decision tree classifier
    
    Attributes
    ----------
    is_trained : bool
        Keeps track of whether the classifier has been trained
    
    Methods
    -------
    train(X, y)
        Constructs a decision tree from data X and label y
    predict(X)
        Predicts the class label of samples X
    
    """

    def __init__(self):
        self.is_trained = False
    
    
    def train(self, x, y):
        
        length = len(x)
        alphabet = np.array(["A","C","E","G","O","Q"])
        alphabet_count = np.zeros((len(alphabet)))
        alphabet_probabilty = np.zeros((len(alphabet)))
        information = np.zeros((len(alphabet)))
        
        for indx,letter in enumerate(alphabet):
            for index in range(0,length):
                if(letter == y[index]):
                    alphabet_count[indx]+=1
                
        alphabet_probabilty[:] = np.round(alphabet_count[:]/length,3)
        #print(alphabet_probabilty)
        information[0:] = alphabet_probabilty[0:] * np.log2(alphabet_probabilty[0:])
        #test_info = alphabet_probabilty[0] * np.log2(alphabet_probabilty[0])
        #test_info_2 = alphabet_probabilty[1] * np.log2(alphabet_probabilty[1])
        #print(information)
        root_entropy = -1*np.sum(information)
        #print(root_entropy)
        
        """
        Parameters
        ----------
        x : numpy.array
            An N by K numpy array (N is the number of instances, K is the 
            number of attributes)
        y : numpy.array
            An N-dimensional numpy array
        
        Returns
        -------
        DecisionTreeClassifier
            A copy of the DecisionTreeClassifier instance
        
        """
        
        # Make sure that x and y have the same number of instances
        assert x.shape[0] == len(y), \
            "Training failed. x and y must have the same number of instances."
        
        

        #######################################################################
        #                 ** TASK 2.1: COMPLETE THIS METHOD **
        #######################################################################
        
        
        
        # set a flag so that we know that the classifier has been trained
        self.is_trained = True
        
        return self
    
    
    def predict(self, x):
        
        
        
        
        """ Predicts a set of samples using the trained DecisionTreeClassifier.
        
        Assumes that the DecisionTreeClassifier has already been trained.
        
        Parameters
        ----------
        x : numpy.array
            An N by K numpy array (N is the number of samples, K is the 
            number of attributes)
        
        Returns
        -------
        numpy.array
            An N-dimensional numpy array containing the predicted class label
            for each instance in x
        """
        
        # make sure that classifier has been trained before predicting
        if not self.is_trained:
            raise Exception("Decision Tree classifier has not yet been trained.")
        
        # set up empty N-dimensional vector to store predicted labels 
        # feel free to change this if needed
        predictions = np.zeros((x.shape[0],), dtype=np.object)
        
        
        #######################################################################
        #                 ** TASK 2.2: COMPLETE THIS METHOD **
        #######################################################################
        
    
        # remember to change this if you rename the variable
        return predictions
        

