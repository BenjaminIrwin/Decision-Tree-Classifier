##############################################################################
# CO395: Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the train() and predict() methods of the 
# DecisionTreeClassifier 
##############################################################################

import numpy as np
from numpy import ma


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
    
    
    def train(self, x, y,node = []):
        
        # Make sure that x and y have the same number of instances
        assert x.shape[0] == len(y), \
            "Training failed. x and y must have the same number of instances."
        
        labels = np.unique(y)
        if(len(labels) <= 1):
            
            return
        
        if(len(x) <=1):
            
            return
        
        root_node = find_best_node(x,y) 
        
        #Resort the data according to the correct collumn            
        self.sort_dataset(x,y,length,node[3])
        #split the data set
        subset_1_x = x[:node[2]]
        subset_2_x = x[node[2]:]
        subset_1_y = x[:node[2]]
        subset_2_y = x[node[2]:]
        
        #Recursively call the function on the split dataset
        child_node_1 = train(self,subset_1_x,subset_1_y)
        child_node_2 = train(self,subset_2_x,subset_2_y)
        
        
        
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
        
    def find_probabilitys(self,x,y,class_labels,length):
        
        alphabet_count = np.zeros((len(class_labels)))
        alphabet_probabilty = np.zeros((len(class_labels)))
        
        
        for indx,letter in enumerate(class_labels):
            for index in range(0,length):
                if(letter == y[index]):
                    alphabet_count[indx]+=1
        
        #Binary Classifcation
        alphabet_probabilty[:] = np.round(alphabet_count[:]/length,3)
        
        return alphabet_probabilty
    
    def find_entropy(self,alphabet,probablity_matrix):
        
        information = np.zeros((len(alphabet)))
                
        ## need to deal with zeros
        
        information = probablity_matrix * ma.log2(probablity_matrix)
        root_entropy = -1*np.sum(information)
        
        return root_entropy
    
#    def induce_decision_tree(self,x,y,width,length,root_entropy)
    def sort_dataset(self,x,y,length,col):
        
        for j in range(0,length):
            for j_2 in range(j,length):
                if(x[j_2][col] < x[j][col]):
                    
                    #swapping 
                    temp = x[j_2][:]
                    temp_y = y[j_2]
                    x[j_2][:] = x[j][:]
                    x[j][:] = temp
                    y[j_2] = y[j]
                    y[j] = temp_y
                    
                
    def find_best_node(self,x,y):
        
        [length,width] = np.shape(x)
        alphabet = np.array(["A","C","E","G","O","Q"])
        alphabet_probabilty = self.find_probabilitys(x,y,alphabet,length)
        
        information = np.zeros((len(alphabet)))
        information[0:] = alphabet_probabilty[0:] * np.log2(alphabet_probabilty[0:])

        root_entropy = self.find_entropy(alphabet,alphabet_probabilty)
        previous_gain = 0
        stored_x= np.zeros((0,6))
        stored_y= 0
        stored_row = 0
        stored_col = 0
        previous_gain = root_entropy
        
        for collumn in range(0,width):
            
            self.sort_dataset(x,y,length,collumn)
            
            for row in range(1,length):
                
                #split the attributes if the letter is not the same
                if(y[row] != y[row-1]):
                    
                    subset_1 = x[:row]
                    subset_2 = x[row:]
                    
                    #get the probabliltys of each set
                    subset_1_prob = self.find_probabilitys(subset_1,y,alphabet,len(subset_1))
                    subset_2_prob = self.find_probabilitys(subset_2,y,alphabet,len(subset_2))

                    #get entropys of each set
                    subset_1_entropy = self.find_entropy(alphabet,subset_1_prob)
                    subset_2_entropy = self.find_entropy(alphabet,subset_2_prob)

                    subset_1_entropy_normalised = subset_1_entropy * len(subset_1)/length
                    subset_2_entropy_normalised = subset_2_entropy * len(subset_1)/length
                    
                    #get total entropy
                    total_entropy = subset_1_entropy_normalised + subset_2_entropy_normalised
                    
                    #get Gain
                    gain = root_entropy - total_entropy
                    #check whether it is bigger than the previous
                    if(gain > previous_gain):
                        stored_x = x[row][:]
                        stored_y = y[row]
                        stored_row = row
                        stored_col = collumn
                    
                    
                    # if not keep going 
                    # if so store the row and collumn and keep going
                    previous_gain = gain
        
        #returns the node
        return (stored_x,stored_y,stored_row,stored_col)
    
    
    def terminal_leaf(self,data_set):
        labels,count = np.unique(data_set,return_counts = True)
        index = np.argmax(count)
        return data_set[index]
        
        
        
        
            
            
            
            
