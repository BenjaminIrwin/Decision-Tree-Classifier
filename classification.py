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
import copy

from eval import Evaluator

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

    def load_data(self, filename):

        data_set = np.loadtxt(filename, dtype=str, delimiter=',')
        num_samp = len(data_set) #number of sample_is
        num_att = len(data_set[0]) #number of attributes
        
        #create attribute and label arrays filled with zeros
        x = np.zeros((num_samp, num_att - 1))
        y = np.zeros(num_samp, dtype=str)
        
        #fill arrays with correct values
        for sample_i in range(num_samp):
            for attribute_i in range(num_att):
                if attribute_i < (num_att - 1):
                    x[sample_i][attribute_i] = data_set[sample_i][attribute_i]
                else:
                    y[j][0] = data_set[j][i]
        return x,y


    def evaluate_input(self,x,y):

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
        '''''
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

        # Make sure that x and y have the same number of instances
        assert x.shape[0] == len(y), \
            "Training failed. x and y must have the same number of instances."

        self.root_node = self.induce_decision_tree(x, y)
        # self.simple_node = self.induce_decision_tree(x,y,True)
        
        #print(self.root_node)
        # set a flag so that we know that the classifier has been trained
        self.is_trained = True

        return self

    def induce_decision_tree(self, x, y, optimsed=False):

        # Check whether they all equal the same thing
        labels = np.unique(y)
        length = len(labels)
        if length == 1:
            return labels[0]

        # Nothing in the data set
        if len(x) == 0:
            return None

        if not optimsed:
            node = self.find_best_node_ideal(x, y)
        else:
            node = self.find_best_node_simple(x, y)

        child_1, child_2 = self.split_dataset(node)

        node["majority_class"] = self.most_common_label(node["data"]["labels"])

        # dont need the data after it is split and majority label found
        del (node["data"])

        if len(child_1["attributes"]) == 0 or len(child_2["attributes"]) == 0:
            #recombine array of labels (one will be empty)
            whole_set = np.concatenate((child_1["labels"], child_2["labels"]), axis=0)
            #return most common label as terminal leaf node
            return self.most_common_label(whole_set)

        #Recursively call the function on the split dataset
        node["left"] = self.induce_decision_tree(child_1["attributes"], child_1["outcomes"], optimsed)
        node["right"] = self.induce_decision_tree(child_2["attributes"], child_2["outcomes"], optimsed)

        return node

    # Finds the propabilty of the class lables in the data set
    def find_probabilitys(self, x, y, class_labels, length):

        alphabet, alphabet_count = np.unique(y, return_counts=True)
        alphabet_count = np.array(alphabet_count, dtype=float)

        #Binary Classifcation
        alphabet_probabilty = alphabet_count / length

        return alphabet_probabilty

    # Finds the entropy of the dataset
    def find_entropy(self, class_labels, probablity_matrix):

        information = np.zeros((len(class_labels)))
        information = probablity_matrix * ma.log2(probablity_matrix)
        information[information.mask] = 0
        entropy = -1 * np.sum(information)
        return entropy

    def terminal_leaf(self, data_set):
        labels, count = np.unique(data_set, return_counts=True)
        index = np.argmax(count)
        #print("FROM HERE!!!!")
        #print(data_set)
        return str(data_set[index])

    def find_best_node_ideal(self, x, y):

        length, width = np.shape(x)
        class_labels = np.unique(y)

        class_labels_probabilty = self.find_probabilitys(x, y, class_labels, length)

        root_entropy = self.find_entropy(class_labels, class_labels_probabilty)
        previous_gain = 0
        stored_value = 0
        stored_attribute = 0

        for attribute in range(0, width):

            minimum_attribute_value = int(np.amin(x[:, attribute]))
            maximum_attribute_value = int(np.amax(x[:, attribute]))

            for split_value in range(minimum_attribute_value, maximum_attribute_value+1):

                subset_1_x = []
                subset_1_y = []
                subset_2_x = []
                subset_2_y = []

                for row in range(1, length):

                    if x[row][attribute] < split_value:

                        subset_1_x.append(x[row][:])
                        subset_1_y.append(y[row][0])
                    else:
                        subset_2_x.append(x[row][:])
                        subset_2_y.append(y[row][0])

                outcomes_1 = np.unique(subset_1_y)
                outcomes_2 = np.unique(subset_2_y)

                # get the probabliltys of each set
                subset_1_prob = self.find_probabilitys(subset_1_x, subset_1_y, outcomes_1, len(subset_1_x))
                subset_2_prob = self.find_probabilitys(subset_2_x, subset_2_y, outcomes_2, len(subset_2_y))

                if len(subset_1_prob) != 0:
                    # get entropys of each set
                    subset_1_entropy = self.find_entropy(outcomes_1, subset_1_prob)
                else:
                    subset_1_entropy = 0

                if len(subset_2_prob) != 0:
                    subset_2_entropy = self.find_entropy(outcomes_2, subset_2_prob)
                else:
                    subset_2_entropy = 0

                subset_1_entropy_normalised = subset_1_entropy * len(subset_1_x) / length
                subset_2_entropy_normalised = subset_2_entropy * len(subset_2_x) / length

                # get total entropy
                total_entropy = subset_1_entropy_normalised + subset_2_entropy_normalised

                # get gain
                gain = root_entropy - total_entropy

                # check whether it is bigger than the previous
                if (gain > previous_gain):
                    stored_attribute = attribute
                    stored_value = split_value
                    stored_gain = gain
                    # if not keep going 
                    # if so store the row and collumn and keep going
                    previous_gain = gain

        # get the data into the node here
        data = {"attributes": x, "outcomes": y}

        # returns the node
        return {"value": stored_value, "attribute": stored_attribute,
               "gain": best_gain, "data": data, "left": None,\
                    "right": None, "majority_class": None, "is_checked": False}

    def split_dataset(self, node):

        dataset = node["data"]
        x = dataset["attributes"]
        y = dataset["outcomes"]
        attribute = node["attribute"]
        split_value = node["value"]
        left_x = []
        right_x = []
        left_y = []
        right_y = []

        for row in range(0, len(x)):

            if x[row][attribute] < split_value:
                left_x.append(x[row][:])
                left_y.append(y[row][0])
            else:
                right_x.append(x[row][:])
                right_y.append(y[row][0])

        left = {"attributes": np.array(left_x), "outcomes": np.array(left_y)}
        right = {"attributes": np.array(right_x), "outcomes": np.array(right_y)}

        return left, right

    def predict(self, x, other_tree = False, tree = None):
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
        predictions = np.zeros(x.shape[0], dtype=str)

        # load the classifier
        if other_tree:
            root_of_tree = tree
        else:
            root_of_tree = self.root_node

        for j in range(0, len(x)):
            predictions[j] = self.recursive_predict(root_of_tree, x[j][:])

        # remember to change this if you rename the variable
        return predictions

    def recursive_predict(self, tree, attributes):

        # Check the required attribute is greater or less than the node split
        # then recursively call function on tree from child node.
        #print(type(tree))
     
        if attributes[tree["attribute"]] < tree["value"]:
            return self.recursive_predict(tree["left"], attributes)
        else:
            return self.recursive_predict(tree["right"], attributes)

            if isinstance(tree["left"], dict):
                return self.recursive_predict(tree["left"], attributes)
            else:
                return tree["left"]

        else:
            if isinstance(tree["right"], dict):
                return self.recursive_predict(tree["right"], attributes)
            else:
                return tree["right"]


    
    def node_height(self,node):
        
        if not isinstance(node, dict):
            return 0
        
        return 1 + max(self.node_height(node["left"]),self.node_height(node["right"]))


    def prune_wrapper(self, tree, v_filename):
        """
        Wrapper function to load data from file into x and y numpy array
        format and feed into prune_tree_simple
        Args:
            tree (dict) - tree to be pruned
            v_filename (str) - name of file containg data to validate pruning.
        Output:
            tree (dict or str) - tree pruned such that any additional pruning
                would lower predictive accuracy on validation set.
        """

        x, y = self.load_data(v_filename)

        return self.prune_tree_simple(tree, x, y)

    def prune_tree_simple(self, tree, x_val, y_val):
        """
        Function to accept prunes which increase the tree's accuracy, otherwise 
        ignore
        Args:
            tree (dict) - tree to be pruned
            x_val (2D array) - 2D array of attributes of validation set where
                each row is a differnt sample and each column is a differnt 
                attribute
            y_val (1D array) - 1D array of correct labels for x_val validation 
                data
        Output:
            tree (dict or str) - tree pruned such that any additional pruning
                would lower predictive accuracy on validation set.
        """
        predictions = self.predict(x_val)
        eval = Evaluator()
        confusion = eval.confusion_matrix(predictions, y_val)
        root_accuracy = eval.accuracy(confusion)
        print("Original Accuracy: ", root_accuracy)

        is_pruned = True
        while (is_pruned and isinstance(tree, dict)):
            #make copy of tree then attempt to prune copy
            tree_copy = copy.deepcopy(tree)
            (is_pruned, tree_copy, tree) = self.prune(tree_copy, tree)
            if is_pruned:
                #compare accuracy of pruned tree to original
                new_predictions = self.predict(x_val, True, tree_copy)
                new_confusion = eval.confusion_matrix(new_predictions, y_val)
                new_accuracy = eval.accuracy(new_confusion)
                if new_accuracy >= root_accuracy:
                    #if greater or equal accuracy make tree = copy
                    root_accuracy = new_accuracy
                    tree = copy.deepcopy(tree_copy)
        
        print("New Accuracy: ", root_accuracy)
        return tree




    def prune(self, tree_copy, tree):
        """
        Recursive function to replace first node with two leaves as the 
        majority class of that node. It will only do this if the node has not
        already been checked by the outer algorithm.
        Args:
            tree (dict or str) - current tree
            tree_copy (dict or str) - copy of tree
        Output:
            is_pruned, tree_copy, tree
            
            is_pruned (bool) - was a node found that could be pruned
            tree_copy (dict or str) - pruned tree
            tree (dict or str) - unpruned tree with node which was pruned in
                tree_copy marked as checked: tree["is_checked"] = True
        """
        is_left_leaf = isinstance(tree["left"], str)
        is_right_leaf = isinstance(tree["right"], str)

        #if both children leaves
        if is_left_leaf and is_right_leaf and not tree["is_checked"]:
            tree_copy = copy.deepcopy(tree_copy["majority_class"])
            tree["is_checked"] = True
            return True, tree_copy, tree

        #if left not leaf
        if not is_left_leaf:
            branch_a = self.prune(tree_copy["left"], tree["left"])
            #save updated trees
            tree_copy["left"] = copy.deepcopy(branch_a[1])
            tree["left"] = copy.deepcopy(branch_a[2])
            if branch_a[0]:
                #return tree if pruning done otherwise try right branch
                return True, tree_copy, tree

        #if right not leaf
        if not is_right_leaf:
            branch_a = self.prune(tree_copy["right"], tree["right"])
            #save updated tree
            tree_copy["right"] = copy.deepcopy(branch_a[1])
            tree["right"] = copy.deepcopy(branch_a[2])
            if branch_a[0]:
                #return tree if pruning done
                return True, tree_copy, tree

        return False, tree_copy, tree

    
    def count_leaves(self, tree, count = 0):
        """
        Recursive function to count number of leaves in a tree
        Args:
            tree (dict) - tree to test
            count (int) - current count for number of leaves (used in recusive
                  call)
        Output:
            count (int) - number of leaves in tree
        """

        is_left_leaf = isinstance(tree["left"], str)
        is_right_leaf = isinstance(tree["right"], str)

        #if both children leaves
        if is_left_leaf and is_right_leaf:
            count +=2
            return count

        #if left not leaf
        if  not is_left_leaf:
            if is_right_leaf:
                count += 1  
            count = self.count_leaves(tree["left"], count)

        #if right not leaf
        if not is_right_leaf:
            if is_left_leaf:
                count += 1
            count = self.count_leaves(tree["right"], count)
            
        return count




    def print_tree(self,tree):
        
        #Attrubute column labels
        attributes = {0:"x-box",1:"y-box",2:"width",3:"high",
                      4:"onpix",5:"x-bar",6:"y-bar",7:"x2bar",
                      8:"y2bar",9:"xybar",10:"x2ybr",11:"xy2br",12:"x-ege",13:"xegvy",14:"y-ege",15:"yegvx"}
        
        fig,ax = plt.subplots(nrows = 1,ncols=1)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        y = 100000000
        x1 = 0
        x2 = 100000000
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
#        if depth ==3:
#            return
    
        #else do all this stuff
        annotation = "depth:"+str(0) + " " + attributes[node["attribute"]] + "<" + str(node["value"])
        
        left_height = self.node_height(node["left"]) + 1
        right_height = self.node_height(node["right"]) + 1
        weight = left_height/(left_height + right_height)
        
        weighted_x = x1 + weight*(x2-x1)
    
        self.recursive_print(node["left"],mid_x,x1,weighted_x,y-2*height,attributes,depth+1,patches,ax)
        self.recursive_print(node["right"],mid_x,weighted_x,x2,y-2*height,attributes,depth+1,patches,ax)
