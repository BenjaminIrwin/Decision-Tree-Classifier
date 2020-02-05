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
from matplotlib.collections import PatchCollection


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
        self.root_node = {}

    def load_data(self, filename):
        """
        Function to load data from file
        Args:
            filename (str) - name of .txt file you are loading data from
        Output:
            (x, y) (tuple) - x: 2D array of training data where each row
            corresponds to a different sample and each column corresponds to a
            different attribute.
                            y: 1D array where each index corresponds to the
            ground truth label of the sample x[index][]
        """
        #load data to single 2D array
        data_set = np.loadtxt(filename, dtype=str, delimiter=',')
        num_samp = len(data_set) #number of sample_is
        num_att = len(data_set[0]) #number of attributes
        
        #create attribute and label arrays filled with zeros
        x = np.zeros((num_samp, num_att - 1))
        y = np.zeros((num_samp,1), dtype=str)
        
        #fill arrays with correct values
        for sample_i in range(num_samp):
            for attribute_i in range(num_att):
                if attribute_i < (num_att - 1):
                    x[sample_i][attribute_i] = data_set[sample_i][attribute_i]
                else:
                    y[sample_i] = data_set[sample_i][attribute_i]
        return x, y


    def evaluate_input(self, x, y):
        """
        Function to evaluate data loaded from file
        Args:
            x (2D array) - 2D array of training data where each row
            corresponds to a different sample and each column corresponds to a
            different attribute.
            y (1D array) -  where each index corresponds to the
            ground truth label of the sample x[index][]
        """
        #these can be shown graphically
        alphabet, count = np.unique(y, return_counts=True)
        alphabet_count = np.zeros((len(alphabet)))
        alphabet_proportions_1 = count / len(y)
        print("alphabet:")
        print(alphabet)
        print("alphabet proportions:")
        print(alphabet_proportions_1)

        length, width = np.shape(x)
        print('test')
        print(np.amax(x[:, 0]))
        minimum_attribute_value = np.amin(x, axis=0)
        maximum_attribute_value = np.amax(x, axis=0)
        attribute_ranges_1 = maximum_attribute_value - minimum_attribute_value
        print("minimum:")
        print(minimum_attribute_value)
        print("maximum:")
        print(maximum_attribute_value)
        print("attribute ranges:")
        print(attribute_ranges_1)
        '''
        filename = "data/train_noisy.txt"
        classifier = DecisionTreeClassifier()
        x,y = classifier.load_data(filename)

        print(len(x))
        print(len(x[0,:]))

        alphabet,count = np.unique(y,return_counts=True)
        alphabet_count = np.zeros((len(alphabet)))
        alphabet_proportions = count/len(y)
        print("alphabet:")
        print(alphabet)
        print("alphabet proportions:")
        print(alphabet_proportions)

        length,width = np.shape(x)

        minimum_attribute_value = np.amin(x,axis=0)
        maximum_attribute_value = np.amax(x,axis=0)
        attribute_ranges = maximum_attribute_value - minimum_attribute_value
        print("minimum:")
        print(minimum_attribute_value)
        print("maximum:")
        print(maximum_attribute_value)
        print("attribute ranges:")
        print(attribute_ranges)

        range_difference = attribute_ranges_1-attribute_ranges
        proportion_difference = alphabet_proportions_1-alphabet_proportions
        print(range_difference)
        print(np.round(proportion_difference*100,2))
        '''

    def train(self, x, y):
        """
        Function to creat decision tree based on training data in x, y
        Args:
            x (2D array) - 2D array of training data where each row
            corresponds to a different sample and each column corresponds to a
            different attribute.
            y (1D array) -  where each index corresponds to the
            ground truth label of the sample x[index][]

        Output:
            self
        """
        # Make sure that x and y have the same number of instances
        assert x.shape[0] == len(y), \
            "Training failed. x and y must have the same number of instances."

        self.root_node = self.induce_decision_tree(x, y)
        # self.simple_node = self.induce_decision_tree(x,y,True)
        
        print(self.root_node)
        # set a flag so that we know that the classifier has been trained
        self.is_trained = True

        return self

    def induce_decision_tree(self, x, y):
        """
        Recursive function to create decision tree based on training data
        in x, y
        Args:
            x (2D array) - 2D array of training data where each row
            corresponds to a different sample and each column corresponds to a
            different attribute.
            y (1D array) -  where each index corresponds to the
            ground truth label of the sample x[index][]
        Output:
            node (dict) - root node of decision tree with child nodes stored
            under "left" and "right" keys. The leaf nodes contain a string
            corresponding to the label.
        """

        labels = np.unique(y) #return array of unique labels
        length = len(labels) #number of unique labels

        """
        When all data in x corresponds to 1 type of label, no further 
        partitioning is needed. Hence return label as leaf node. 
        """
        if length == 1:
            return labels[0]

        # Nothing in the data set
        if len(x) == 0:
            print("induce_decision_tree Error: No sample data passed into "
                  "function.")
            return None

        #find best partition for node
        node = self.find_best_node_ideal(x, y)

        #divide data on found partition
        child_1, child_2 = self.split_dataset(node)

        # dont need the data after it is split
        del (node["data"])

        """
        When there are still multiple different types of labels in data but any
        kind of division does not decrease entropy ('best' information gain 
        with no division) a leaf node has been found -- terminate recursion
        and choose most common label in data as leaf label.
        """
        if len(child_1["attributes"]) == 0 or len(child_2["attributes"]) == 0:
            #recombine array of labels (one will be empty)
            whole_set = np.concatenate((child_1["labels"], child_2["labels"]))
            #return most common label as terminal leaf node
            return self.most_common_label(whole_set)

        # Recursively call the function on the split dataset
        node["left"] = self.induce_decision_tree(child_1["attributes"],
                                                 child_1["labels"])

        node["right"] = self.induce_decision_tree(child_2["attributes"],
                                                  child_2["labels"])

        return node

    def find_total_entropy(self, y):
        """
        Function to find the total entropy in label array, y
        Args:
            y (1D array) -  where each index corresponds to the
            ground truth label of the sample x[index][]
        Output:
            entropy (float) - calculated entropy
        """
        num_samples = len(y)
        if num_samples == 0:
            return 0
        #find probabilities of each label
        labels_unique, label_count = np.unique(y, return_counts=True)
        label_probabilities = label_count / num_samples
        #find entropy using probabilities
        information = label_probabilities * np.log2(label_probabilities)
        entropy = -1 * np.sum(information)
        return entropy

    def most_common_label(self, data_set):
        """
        Returns the most frequent value in 1D numpy array
        Args:
            data_set (1D array)
        Output:
           most common value in array
        """
        labels, count = np.unique(data_set, return_counts=True)
        index = np.argmax(count)
        return data_set[index]

    def find_best_node_ideal(self, x, y):
        """
        Function to find the attribute and value on which a binary partition of
        the data can be made to maximise information gain (entropy reduction)
        Args:
            x (2D array) - 2D array of training data where each row
            corresponds to a different sample and each column corresponds to a
            different attribute.
            y (1D array) -  where each index corresponds to the
            ground truth label of the sample x[index][]
        Output:
            node (dict) - dictionary contains information on partition,
            including value and attribute to partition over.
        """
        stored_value = 0
        stored_attribute = 0
        best_gain = 0

        num_samples, num_attributes = np.shape(x)

        root_entropy = self.find_total_entropy(y)

        for attribute in range(num_attributes):
            """
            iterate through each attribute to find which attribute to 
            split data on --> find which attribute partition causes the
            greatest information gain.
            """
            #find the min and max value of that attribute
            minimum_attribute_value = int(np.amin(x[:, attribute]))
            maximum_attribute_value = int(np.amax(x[:, attribute]))

            for split_value in range(minimum_attribute_value,
                                     maximum_attribute_value+1):
                """
                iterate through each possible divide of that attribute 
                (where to halve data) to find what value of that attribute to 
                split data on --> find which partition causes the greatest 
                information gain.
                """
                #first half
                subset_1_x = []
                subset_1_y = []

                #second half
                subset_2_x = []
                subset_2_y = []

                # WHY DO YOU START FROM 1 HER
                for row in range(num_samples):

                    #perform separation of data into two halves
                    if x[row][attribute] < split_value:
                        subset_1_x.append(x[row][:])
                        subset_1_y.append(y[row])
                    else:
                        subset_2_x.append(x[row][:])
                        subset_2_y.append(y[row])

                #find entropy of each half
                subset_1_entropy = self.find_total_entropy(subset_1_y)
                subset_2_entropy = self.find_total_entropy(subset_2_y)

                #normalise entropy for each of sub datasets
                subset_1_entropy_normalised = \
                    subset_1_entropy * len(subset_1_x) / num_samples
                subset_2_entropy_normalised = \
                    subset_2_entropy * len(subset_2_x) / num_samples

                # get total entropy
                total_split_entropy = subset_1_entropy_normalised + \
                                      subset_2_entropy_normalised

                # get information gain
                information_gain = root_entropy - total_split_entropy

                # check whether it is bigger than the previous
                if (information_gain > best_gain):
                    stored_attribute = attribute
                    stored_value = split_value
                    best_gain = information_gain

        # get the data into the node here
        data = {"attributes": x, "labels": y}

        # returns the node
        return {"value": stored_value, "attribute": stored_attribute,
               "gain": best_gain, "data": data, "left": None,\
                    "right": None}

    def split_dataset(self, node):
        """
        Function to split the data in a node according to partition defined by
        find_best_node_ideal
        Args:
            node (dict) - node which details how to split data
        Output:
            (left, right) (tuple) - dataset split into two halves as defined by
            node["attribute"] and node["value"]
        """
        dataset = node["data"]
        x = dataset["attributes"]
        y = dataset["labels"]
        attribute = node["attribute"]
        split_value = node["value"]
        left_x = []
        right_x = []
        left_y = []
        right_y = []

        for row in range(len(x)):

            if x[row][attribute] < split_value:
                left_x.append(x[row][:])
                left_y.append(y[row][0])
            else:
                right_x.append(x[row][:])
                right_y.append(y[row][0])

        left = {"attributes": np.array(left_x), "labels": np.array(left_y)}
        right = {"attributes": np.array(right_x), "labels": np.array(right_y)}

        return left, right

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
            raise Exception("Decision Tree classifier has not yet been"
                            " trained.")

        # set up empty N-dimensional vector to store predicted labels 
        predictions = np.zeros((x.shape[0],), dtype=str)
        # load the classifier
        root_of_tree = self.root_node

        for j in range(len(x)):
            predictions[j] = self.recursive_predict(root_of_tree, x[j][:])

        # remember to change this if you rename the variable
        return predictions

    def recursive_predict(self, tree, attributes):
        """
        Function to predict the label of a sample based on its attributes
        Args:
            tree (dict) - trained decision tree
            attributes (2D array) - 2D array of test data where each row
            corresponds to a different sample and each column corresponds to a
            different attribute.
        Output:
            string object of label prediction
        """
        #if leaf found return label str
        if isinstance(tree, str):
            return tree

        # Check the required attribute is greater or less than the node split
        # then recursively call function on tree from child node.
        elif attributes[tree["attribute"]] < tree["value"]:
            return self.recursive_predict(tree["left"], attributes)
        else:
            return self.recursive_predict(tree["right"], attributes)




    
    def node_height(self,node):
        
        if not isinstance(node, dict):
            return 0
        
        return 1 + max(self.node_height(node["left"]),self.node_height(node["right"]))

                                         
    
    def print_tree(self,tree):
        
        #Attrubute column labels
        attributes = {0:"x-box",1:"y-box",2:"width",3:"high",
                      4:"onpix",5:"x-bar",6:"y-bar",7:"x2bar",
                      8:"y2bar",9:"xybar",10:"x2ybr",11:"xy2br",12:"x-ege",13:"xegvy",14:"y-ege",15:"yegvx"}
        
        fig,ax = plt.subplots(nrows = 1,ncols=1)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        y = 1000
        x1 = 0
        x2 = 1000
        mid_x = (x1 + x2)/2
        height = 50
        width = 200
        depth = 0
        
        patches = []
        patches.append(pp.Rectangle((mid_x-width/2,y-height),width,height,color = 'blue'))
        annotation = "Node:\nDepth:"+str(0) + "\nAttribute Split: " + attributes[tree["attribute"]] + "<" + str(tree["value"]) + "\nInformation Gain:"+str(np.round(tree["gain"],3))
        center_x = mid_x #(mid_x-width/2)
        center_y = y - height/2.0
        ax.annotate(annotation, (center_x,center_y), color='white', weight='bold',fontsize=6, ha='center', va='center')
        
        self.recursive_print(tree["left"],mid_x,x1,mid_x,y-2*height,attributes,depth+1,patches,ax)
        self.recursive_print(tree["right"],mid_x,mid_x,x2,y-2*height,attributes,depth+1,patches,ax)
        
        ax.add_collection(PatchCollection(patches,match_original=True))
        ax.set_xlim((0,1000))
        ax.set_ylim((0,1000))
        ax.autoscale()
        plt.show()                     
        
        
    def recursive_print(self,node,parent_center_x,x1,x2,y,attributes,depth,patches,ax):
        
        mid_x = (x1 + x2)/2
        height = 50
        width = 200

        if not isinstance(node, dict):
            #print a leaf node (different colour
    
            patches.append(pp.Rectangle((mid_x-width/2,y-height),width,height,color = 'black'))
            annotation = "Leaf Node \nLabel = " + str(node)
            center_x = mid_x
            center_y = y - height/2.0
            ax.annotate(annotation, (center_x, center_y), color='white', weight='bold',fontsize=6, ha='center', va='center')
            plt.plot([parent_center_x,mid_x],[y+height,y],'black',linestyle=':',marker='')
            return 
        
        else:
            
            patches.append(pp.Rectangle((mid_x-width/2,y-height),width,height,color = 'blue'))
            annotation = "Node:\nDepth:"+str(depth) + "\nAttribute Split: " + attributes[node["attribute"]] + "<" + str(node["value"])+ "\nInformation Gain:"+str(np.round(node["gain"],3))
            center_x = mid_x
            center_y = y - height/2.0
            ax.annotate(annotation, (center_x,center_y), color='white', weight='bold',fontsize=6, ha='center', va='center')
            plt.plot([parent_center_x,mid_x],[y+height,y],'black',linestyle=':',marker='')
        
        #Maximum depth to print out
        if depth ==3:
            return
    
        #else do all this stuff
        annotation = "depth:"+str(0) + " " + attributes[node["attribute"]] + "<" + str(node["value"])
        
        left_height = self.node_height(node["left"]) + 1
        right_height = self.node_height(node["right"]) + 1
        weight = left_height/(left_height + right_height)
        
        weighted_x = x1 + weight*(x2-x1)
    
        self.recursive_print(node["left"],mid_x,x1,weighted_x,y-2*height,attributes,depth+1,patches,ax)
        self.recursive_print(node["right"],mid_x,weighted_x,x2,y-2*height,attributes,depth+1,patches,ax)
        
    
  
       
