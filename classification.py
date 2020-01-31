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
import matplotlib.pyplot as plt
import matplotlib.patches as pp
import matplotlib.collections as mcollections

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
        
    def load_data(self,filename):
    
        data_set= np.loadtxt(filename,dtype=str,delimiter=',')
        length = len(data_set)
        line_length = len(data_set[0])
        x = np.zeros((length,line_length-1))
        y = np.zeros((length,1),dtype = str)
        
        for j in range(0,length):
            for i in range(0,line_length):
                if i < (line_length-1):
                    x[j][i] = data_set[j][i]
                else:
                    y[j][0] = data_set[j][i]
        return x,y
    
    
    
    def train(self, x, y):
        
        # Make sure that x and y have the same number of instances
        assert x.shape[0] == len(y), \
            "Training failed. x and y must have the same number of instances."
        
        self.root_node = self.induce_decision_tree(x,y)
        #self.simple_node = self.induce_decision_tree(x,y,True)
        
        # set a flag so that we know that the classifier has been trained
        self.is_trained = True
        
        return self
    
    
    
    def induce_decision_tree(self,x,y,optimsed = False):
        
        #Check whether they all equal the same thing
        labels = np.unique(y)
        length = len(labels)
        if length == 1:
            return labels[0]
        
        #Nothing in the data set
        if len(x) == 0:
            return None

        if optimsed == False:
            node = self.find_best_node_ideal(x,y)
        else:
            node = self.find_best_node_simple(x,y)
            
        child_1,child_2 = self.split_dataset(node)
        
        #dont need the data after it is split
        del(node["data"])
        
        if(len(child_1["attributes"]) == 0 or len(child_2["attributes"]) == 0):
            whole_set = np.concatenate((child_1["outcomes"],child_2["outcomes"]), axis=0)
            letter = self.terminal_leaf(whole_set)
            return letter
        
        #Recursively call the function on the split dataset
        node["left"] = self.induce_decision_tree(child_1["attributes"],child_1["outcomes"],optimsed)
        node["right"] = self.induce_decision_tree(child_2["attributes"],child_2["outcomes"],optimsed)
        
        return node
        
    #Finds the propabilty of the class lables in the data set 
    def find_probabilitys(self,x,y,class_labels,length):
        
        alphabet,alphabet_count = np.unique(y,return_counts = True)
        alphabet_count = np.array(alphabet_count,dtype = float)
        
        #Binary Classifcation
        alphabet_probabilty = alphabet_count/length
    
        return alphabet_probabilty
    
    
    #Finds the entropy of the dataset
    def find_entropy(self,class_labels,probablity_matrix):
        
        information = np.zeros((len(class_labels)))
                
        ## need to deal with zeros
        information = probablity_matrix * ma.log2(probablity_matrix)
        information[information.mask] = 0
        root_entropy = -1*np.sum(information)

        return root_entropy
    

    def sort_dataset(self,x,y,col):
        
        length = len(y)
        for j in range(0,length):
            for j_2 in range(j+1,length):
                if(x[j_2][col] < x[j][col]):
                    
                    #swapping 
                    temp = x[j][:]
                    temp_y = y[j]
                    x[j][:] = x[j_2][:]
                    y[j] = y[j_2]
                    x[j_2][:] = temp
                    y[j_2] = temp_y
                    
                    
    def terminal_leaf(self,data_set):
        labels,count = np.unique(data_set,return_counts = True)
        index = np.argmax(count)
        return (data_set[index])    
    
    def find_best_node_ideal(self,x,y):
    
        length,width = np.shape(x)
        class_labels = np.unique(y)

        class_labels_probabilty = self.find_probabilitys(x,y,class_labels,length)

        root_entropy = self.find_entropy(class_labels,class_labels_probabilty)
        previous_gain = 0
        stored_col = 0
        stored_value = 0
        stored_attribute = 0
        #split_value = 5

        for attribute in range(0,width):
            
            for split_value in range(1,16):
            
                subset_1_x = []
                subset_1_y = []
                subset_2_x = []
                subset_2_y = []
                
                for row in range(1,length):
                    
                    if x[row][attribute] < split_value:
                        
                        subset_1_x.append(x[row][:])
                        subset_1_y.append(y[row][0])
                    else:
                        subset_2_x.append(x[row][:])
                        subset_2_y.append(y[row][0])

                outcomes_1 = np.unique(subset_1_y)
                outcomes_2 = np.unique(subset_2_y)

                #get the probabliltys of each set
                subset_1_prob = self.find_probabilitys(subset_1_x,subset_1_y,outcomes_1,len(subset_1_x))
                subset_2_prob = self.find_probabilitys(subset_2_x,subset_2_y,outcomes_2,len(subset_2_y))
                
                
                if len(subset_1_prob) != 0:
                    #get entropys of each set
                    subset_1_entropy = self.find_entropy(outcomes_1,subset_1_prob)
                else:
                    subset_1_entropy = 0
                
                if len(subset_2_prob) != 0:    
                    subset_2_entropy = self.find_entropy(outcomes_2,subset_2_prob)
                else:
                    subset_2_entropy = 0

                subset_1_entropy_normalised = subset_1_entropy * len(subset_1_x)/length
                subset_2_entropy_normalised = subset_2_entropy * len(subset_2_x)/length

                #get total entropy
                total_entropy = subset_1_entropy_normalised + subset_2_entropy_normalised
                
                #get gain
                gain = root_entropy - total_entropy

                #check whether it is bigger than the previous
                if(gain > previous_gain):
                    
                    stored_attribute = attribute
                    stored_value = split_value 
                    stored_gain = gain
                    # if not keep going 
                    # if so store the row and collumn and keep going
                    previous_gain = gain

        #get the data into the node here
        data = {"attributes":x,"outcomes":y}
                
        #returns the node 
        return {"value":stored_value,"attribute":stored_attribute,"gain":stored_gain,"data":data,"left":None ,"right":None}
    
    def split_dataset(self,node):
        
        dataset = node["data"]
        x = dataset["attributes"]
        y = dataset["outcomes"]
        attribute = node["attribute"]
        split_value = node["value"]
        left_x = []
        right_x = []
        left_y = []
        right_y = []
        
        for row in range(0,len(x)):
            
            if x[row][attribute] < split_value:
                left_x.append(x[row][:])
                left_y.append(y[row][0])
            else:
                right_x.append(x[row][:])
                right_y.append(y[row][0])
                
        left = {"attributes":np.array(left_x),"outcomes":np.array(left_y)}
        right = {"attributes":np.array(right_x),"outcomes":np.array(right_y)}
        
        return (left,right)
        

    
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
         #load the classifier
        root_of_tree = self.root_node
        
        for j in range(0,len(x)):
            
        
            predictions[j] = self.recursive_predict(root_of_tree,x[j][:])
    
      
        # remember to change this if you rename the variable
        return predictions
        
    def recursive_predict(self,tree,attributes):
        
        #Check the required attribute is greater or less than the node
        if attributes[tree["attribute"]] < tree["value"]:
            
            if isinstance(tree["left"],dict):
                return self.recursive_predict(tree["left"],attributes)
            else:
                return tree["left"]
            
        else:
            if isinstance(tree["right"],dict):
                return self.recursive_predict(tree["right"],attributes)
            else:
                return tree["right"]
            
            
    def print_tree(self):
        
        tree = self.root_node
        print(tree)
        
        
        #Attrubute column labels
        attributes = {0:"x-box",1:"y-box",2:"width",3:"high",
                      4:"onpix",5:"x-bar",6:"y-bar",7:"x2bar",
                      8:"y2bar",9:"xybar",10:"x2ybr",11:"xy2br",12:"x-ege",13:"xegvy",14:"y-ege",15:"yegvx"}
        
        #recursive_print(tree,attributes,0,patches_2)
        annotation = "depth:"+str(0) + " " + attributes[tree["attribute"]] + "<" + str(tree["value"])
        
        #Code for the root
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        rect = plt.Rectangle((0.45, 0.9), 0.1, 0.1, color='k', alpha=0.3)
        rx, ry = rect.get_xy()
        center_x = rx + rect.get_width()/2.0
        center_y = ry + rect.get_height()/2.0
        
        rectangles = []
        
        rectangles.append(rect)
    
        parent_coordinates = {"xy":(rx,ry),"center_x":center_x,"center_y":center_x}
        
        left_details = []
        right_details = []
        annotation = {}
        center =  {}
        patches_left = []
        patches_right = []
        
        #left branch
        #self.recursive_print(tree,attributes,1,parent_coordinates,patches_left,left_details)
        
        #Right branch
        #self.recursive_print(tree,attributes,1,parent_coordinates,patches_right,right_details)
        

         
        ax.annotate(annotation,(center_x, center_y), color='w', weight='bold',fontsize=6, ha='center', va='center')
        
        
        """
        
        for r in rectangles:
            ax.add_artist(rectangles[r])
            rx, ry = rectangles[r].get_xy()
            cx = rx + rectangles[r].get_width()/2.0
            cy = ry + rectangles[r].get_height()/2.0

            ax.annotate(r, (cx, cy), color='w', weight='bold', 
                fontsize=6, ha='center', va='center')
        """
      
        ax.add_patch(rect)

        #ax.autoscale()

        #rect =  pp.Rectangle((50,100),facecolor='b')
        plt.show()
        
    def recursive_print(self,tree,attributes,depth,parent_coordinates,rectangles,details,left):
        
        if left == True:
            center_x = parent_coordinates["xy"][0]
            center_y = parent_coordinates["xy"][0] - (0.1+0.05)
        
        
        if isinstance(tree,dict):
        
            annotation = "depth:"+str(0) + " " + attributes[tree["attribute"]] + "<" + str(tree["value"])
            
            rectangles.append(plt.Rectangle((0.45, 0.9), 0.1, 0.1, color='k', alpha=0.3)
                              
            center_x = parent_coordinates["xy"][0] #center of the previous
            center_y = parent_coordinates["xy"][0] - 0.1+0.05
            details = {"annotation":annotation,center_x:
                           
        else:
            
            annotation = "depth:"+str(0) + " " + attributes[tree["attribute"]] + "<" + str(tree["value"])
            rect = plt.Rectangle((0.45, 0.9), 0.1, 0.1, color='k', alpha=0.3)
            
            
            
        
        
            
            
            #rect = Rectangle((15,10),10,, angle=0.0,
            
            
        
        
    
        
        
        



























    
    def find_best_node_simple(self,x,y):
    
        length,width = np.shape(x)
        class_labels = np.unique(y)

        class_labels_probabilty = self.find_probabilitys(x,y,class_labels,length)

        root_entropy = self.find_entropy(class_labels,class_labels_probabilty)
        previous_gain = 0
        stored_col = 0
        stored_value = 0
        stored_attribute = 0
        split_value = 0

        for attribute in range(0,1):
            
    
            subset_1_x = []
            subset_1_y = []
            subset_2_x = []
            subset_2_y = []
            
            self.sort_dataset(x,y,attribute)

            for row in range(1,length):
                
                #If the attrbute and outcomes are different
                if x[row][attribute] != x[row-1][attribute] and y[row][0]!= y[row-1][0]:
                    
                    split_value = x[row][attribute]
                    subset_1_x.append(x[:row][:])
                    subset_1_y.append(y[:row][0])
                    subset_2_x.append(x[row][:])
                    subset_2_y.append(y[row][0])
            print(subset_1_x)
            print(subset_1_y)
            print(subset_2_x)
            print(subset_2_y)

            outcomes_1 = np.unique(subset_1_y)
            outcomes_2 = np.unique(subset_2_y)

            #get the probabliltys of each set
            subset_1_prob = self.find_probabilitys(subset_1_x,subset_1_y,outcomes_1,len(subset_1_x))
            subset_2_prob = self.find_probabilitys(subset_2_x,subset_2_y,outcomes_2,len(subset_2_y))
            
            if len(subset_1_prob) != 0:
                #get entropys of each set
                subset_1_entropy = self.find_entropy(outcomes_1,subset_1_prob)
            else:
                subset_1_entropy = 0
            
            if len(subset_2_prob) != 0:    
                subset_2_entropy = self.find_entropy(outcomes_2,subset_2_prob)
            else:
                subset_2_entropy = 0

            subset_1_entropy_normalised = subset_1_entropy * len(subset_1_x)/length
            subset_2_entropy_normalised = subset_2_entropy * len(subset_2_x)/length

            #get total entropy
            total_entropy = subset_1_entropy_normalised + subset_2_entropy_normalised
            
            #get gain
            gain = root_entropy - total_entropy
            

            #check whether it is bigger than the previous
            if(gain > previous_gain):
                stored_attribute = attribute
                stored_value = split_value 
                stored_gain = gain
                # if not keep going 
                # if so store the row and collumn and keep going
                previous_gain = gain
        
        #get the data into the node here
        data = {"attributes":x,"outcomes":y}
      
        #returns the node 
        return {"value":stored_value,"attribute":stored_attribute,"gain":stored_gain,"data":data,"left":None ,"right":None}

        
    def sort_dataset(self,x,y,col):
        
        new_x = np.zeros((len(x),len(x[0,:])))
        new_y = np.zeros((len(y),1))
        
        length = len(y)
        for j in range(0,length):
            for j_2 in range(j+1,length):
                
                
                if(x[j_2][col] < x[j][col]):
                    
                    #swapping
                    temp = np.copy(x[j_2][:], order='K')
                    temp_y = np.copy(y[j_2], order='K')
                    
                    x[j_2][0:] = x[j][0:]
                    x[j][0:] = temp[0:]
                    y[j_2] = y[j]
                    y[j] = temp_y
                    
      
                    
                
        
        
        
            
            
            
            
