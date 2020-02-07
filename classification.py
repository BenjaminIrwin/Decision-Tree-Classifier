##############################################################################
# CO395: Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the train() and predict() methods of the 
# DecisionTreeClassifier 
#http://mlwiki.org/index.php/Cost-Complexity_Pruning
##############################################################################

import numpy as np
from numpy import ma
import matplotlib.pyplot as plt
import matplotlib.patches as pp
from matplotlib.collections import PatchCollection
from tempfile import TemporaryFile
from eval import Evaluator
import copy


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
                    x[sample_i,attribute_i] = data_set[sample_i,attribute_i]
                else:
                    y[sample_i] = data_set[sample_i,attribute_i]
        return x, y
    
    
    def adjacent_values(vals, q1, q3):
        upper_adjacent_value = q3 + (q3 - q1) * 1.5
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
        return lower_adjacent_value, upper_adjacent_value

    def set_axis_style(ax, labels):
        ax.get_xaxis().set_tick_params(direction='out')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_xlim(0.25, len(labels) + 0.75)
        ax.set_xlabel('Sample name')


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

    def train(self, x, y):

        # Make sure that x and y have the same number of instances
        assert x.shape[0] == len(y), \
        "Training failed. x and y must have the same number of instances."

        tree = self.induce_decision_tree(x,y)
        #print(tree)
        np.save('tree.npy',tree)

        # set a flag so that we know that the classifier has been trained
        self.is_trained = True

        return self

    def induce_decision_tree(self, x, y):

        # Check whether they all equal the same thing
        labels = np.unique(y)
        length = len(labels)
        if length == 1:
            return labels[0]

        # Nothing in the data set
        if len(x) == 0:
            return None


        node = self.find_best_node_ideal(x, y)

        child_1, child_2 = self.split_dataset(node)

        if len(child_1["attributes"]) == 0 or len(child_2["attributes"]) == 0:
            whole_set = np.concatenate((child_1["labels"], child_2["labels"]), axis=0)
            return self.most_common_label(data_set)
        
        
        left_probability = len(child_1["labels"])/ len(node["data"])
        right_probability = len(child_2["labels"])/len(node["data"])
      
        parent_labels,parent_count = np.unique(node["data"]["labels"],return_counts=True)
        #node["most_occuring_letter"] = self.terminal_leaf(node["data"]["labels"])
        node["majority_class"] = self.most_common_label(node["data"]["labels"])
        del (node["data"])
        
        left_child_labels = self.count_occurrences(parent_labels,child_1["labels"])
        right_child_labels = self.count_occurrences(parent_labels,child_2["labels"])


        node['K'] = self.compute_k(left_probability,left_child_labels,
                                   right_probability,right_child_labels,parent_count)
        
        
        #Initial Filter to prevent overfitting
        if node['K'] < 15:
           return node["majority_class"]

    
        #Recursively call the function on the split dataset
        node["left"] = self.induce_decision_tree(child_1["attributes"], child_1["labels"])
        node["right"] = self.induce_decision_tree(child_2["attributes"], child_2["labels"])

        return node

    def count_occurrences(self,parent_occurrences,child_data):

        child_occurrences = np.zeros((1,len(parent_occurrences)))

        for i in range(len(parent_occurrences)):
            count = 0
            for j in range(len(child_data)):
                if child_data[j] == parent_occurrences[i]:
                    count+=1
            child_occurrences[0,i] = count

        return child_occurrences

    def compute_k(self,left_probability,left_child_labels,right_probability,
                  right_child_labels,parent_labels):

        K = 0
        for i in range(len(left_child_labels)):
            K += ((left_child_labels[0][i]-(parent_labels[i]*left_probability))**2)/(parent_labels[i]*left_probability)
        for i in range(len(right_child_labels)):
            K += ((right_child_labels[0][i] - (parent_labels[i] * right_probability))**2) / (parent_labels[i] * right_probability)

        return K

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
        return str(data_set[index])


    def terminal_leaf(self, data_set):
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

        data = {"attributes": x, "labels": y}

        #Returns the node
        return {"value": stored_value, "attribute": stored_attribute, "gain": best_gain, "data": data, "left": None,
                "right": None,'K':None,"majority_class": None, "is_checked": False}

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

    def predict(self, x,other_tree = False):

        # make sure that classifier has been trained before predicting
        if not self.is_trained:
            raise Exception("Decision Tree classifier has not yet been trained.")

        # set up empty N-dimensional vector to store predicted labels 
        # feel free to change this if needed
        predictions = np.zeros((x.shape[0],), dtype=np.object)
        
        # load the classifier
        if not other_tree:
            tree = np.load('tree.npy',allow_pickle = True).item()
        else:
            tree = other_tree
        
        for j in range(0, len(x)):
            predictions[j] = self.recursive_predict(tree, x[j][:])

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
        if not isinstance(tree, dict):
            return tree

        # Check the required attribute is greater or less than the node split
        # then recursively call function on tree from child node.
        if attributes[tree["attribute"]] < tree["value"]:
            return self.recursive_predict(tree["left"], attributes)
        
        return self.recursive_predict(tree["right"], attributes)



    def node_height(self,node):

        if not isinstance(node, dict):
            return 0

        return 1 + max(self.node_height(node["left"]),self.node_height(node["right"]))


    def print_tree(self,tree):

        #Attrubute column labels
        attributes = {0:"x-box",1:"y-box",2:"width",3:"high",
                      4:"onpix",5:"x-bar",6:"y-bar",7:"x2bar",
                      8:"y2bar",9:"xybar",10:"x2ybr",11:"xy2br",
                      12:"x-ege",13:"xegvy",14:"y-ege",15:"yegvx"}

        fig,ax = plt.subplots(nrows = 1,ncols=1)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        y = 10000
        x1 = 0
        x2 = 10000
        mid_x = (x1 + x2)/2
        height = 50
        width = 600
        depth = 0

        patches = []
        patches.append(pp.Rectangle((mid_x-width/2,y-height),width,height,
                                    color = 'blue'))
        annotation = "Root:\n" + "\nAtt Split: "+ attributes[tree["attribute"]] 
        annotation += "<" + str(tree["value"]) 
        annotation +=  "\nIG:"+str(np.round(tree["gain"],3))
        center_x = mid_x 
        center_y = y - height/2.0
        ax.annotate(annotation, (center_x,center_y), color='white', 
                    weight='bold',fontsize=4, ha='center', va='center')

        self.recursive_print(tree["left"],mid_x,x1,mid_x,y-2*height,
                             attributes,depth+1,patches,ax)
        self.recursive_print(tree["right"],mid_x,mid_x,x2,y-2*height,
                             attributes,depth+1,patches,ax)

        ax.add_collection(PatchCollection(patches,match_original=True))
        ax.set_xlim((0,1000))
        ax.set_ylim((0,1000))
        ax.autoscale()
        plt.show()


    def recursive_print(self,node,parent_center_x,x1,x2,y,attributes,
                        depth,patches,ax):

        mid_x = (x1 + x2)/2
        height = 50
        width = 600

        if not isinstance(node, dict):
            #print a leaf node (different colour

            patches.append(pp.Rectangle((mid_x-width/2,y-height),width,
                                        height,color = 'black'))
            annotation = "Leaf Node \nLabel = " + str(node)
            center_x = mid_x
            center_y = y - height/2.0
            
            ax.annotate(annotation, (center_x, center_y), color='white', 
                        weight='bold',fontsize=4, ha='center', va='center')
            
            plt.plot([parent_center_x,mid_x],[y+height,y],'black',
                    linestyle=':',marker='')
            return

        else:

            patches.append(pp.Rectangle((mid_x-width/2,y-height),
                                        width,height,color = 'blue'))
            annotation = "IntNode:\n" + "\nAtt Split: "+ attributes[node["attribute"]] 
            annotation += "<" + str(node["value"]) 
            annotation +=  "\nIG:"+str(np.round(node["gain"],3))
            center_x = mid_x
            center_y = y - height/2.0
            ax.annotate(annotation, (center_x,center_y), color='white',
                        weight='bold',fontsize=4, ha='center', va='center')
            plt.plot([parent_center_x,mid_x],[y+height,y],'black',linestyle=':',marker='')

        #Maximum depth to print out taken from Piazza 
        if depth ==3:
            return

        #Create the annotation to place into the center of the rectangle
        annotation = "depth:"+str(0) + " " + attributes[node["attribute"]] 
        annotation += "<" + str(node["value"])
        left_height = self.node_height(node["left"]) + 1
        right_height = self.node_height(node["right"]) + 1
        weight = left_height/(left_height + right_height)

        weighted_x = x1 + weight*(x2-x1)

        self.recursive_print(node["left"],mid_x,x1,weighted_x,
                             y-2*height,attributes,depth+1,patches,ax)
        self.recursive_print(node["right"],mid_x,weighted_x,x2,
                             y-2*height,attributes,depth+1,patches,ax)
        
    
    
    #Attempt to get Cost Complexity Pruning working
    def cost_complexity_pruning(self,node):

        trees = np.array([],dtype=np.object)
        count = 0
        tree_copy = node.copy()
        
        while isinstance(tree_copy['left'],dict) or isinstance(tree_copy['right'],dict):
            
            trees = np.append(trees,self.prune_tree(tree_copy))
            count = count + 1
            tree_copy = copy.deepcopy(tree_copy)
        
        
        return trees
    
    
    def prune_tree(self,node):

        if not isinstance(node['left'],dict) and not isinstance(node['right'],dict):
            return node['majority_class']
        elif isinstance(node['left'],dict):
            node['left'] = self.prune_tree(node['left'])
            return node
        elif isinstance(node['right'], dict):
            node['right'] = self.prune_tree(node['right'])
            return node

        return node
    
    
    def calculate_best_pruned_tree(self,trees,x_test,y_test):
        
        eval = Evaluator()
        stored_j=0
        previous_accuracy = 0
        
        #go through each tree and compute the ratio of caculated error (right/total)
        for j in range(len(trees)):
            
            
            predictions = self.predict(x_test,trees[j])
            confusion = eval.confusion_matrix(predictions, y_test)
            accuracy = eval.accuracy(confusion)
            #print(accuracy)
            if accuracy > previous_accuracy:
                stored_j = j
                previous_accuracy = accuracy
                
        return trees[stored_j],previous_accuracy
            
            
    def get_wrong_prediction_count(predictions,y_test):
        
        count = 0
        for j in range(len(predictions)):
            
            if predictions[j] != y_test[j]:
                count+=1
        
        return count
    
    
    #Nicks code for pruning starts here!! Simple pruninig method
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
                new_predictions = self.predict(x_val,tree_copy)
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
        is_left_leaf = not isinstance(tree["left"], dict)
        is_right_leaf = not isinstance(tree["right"], dict)

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

        is_left_leaf = not isinstance(tree["left"], dict)
        is_right_leaf = not isinstance(tree["right"], dict)

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

    
            
