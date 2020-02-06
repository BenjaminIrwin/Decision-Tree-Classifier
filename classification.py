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
from tempfile import TemporaryFile
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
        length = len(data_set)
        line_length = len(data_set[0])
        x = np.zeros((length, line_length - 1))
        y = np.zeros((length,1), dtype=str)

        for j in range(length):
            for i in range(line_length):
                if i < (line_length - 1):
                    x[j][i] = data_set[j][i]
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

    def train(self, x, y):

        # Make sure that x and y have the same number of instances
        assert x.shape[0] == len(y), \
        "Training failed. x and y must have the same number of instances."

        tree = self.induce_decision_tree(x,y)
        print(tree)
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
            whole_set = np.concatenate((child_1["outcomes"], child_2["outcomes"]), axis=0)
            return self.terminal_leaf(whole_set)
        
        
        left_probability = len(child_1["outcomes"])/ len(node["data"])
        right_probability = len(child_2["outcomes"])/len(node["data"])
      
        parent_labels,parent_count = np.unique(node["data"]["outcomes"],return_counts=True)
        node["most_occuring_letter"] = self.terminal_leaf(node["data"]["outcomes"])
        
        del (node["data"])
        
        left_child_labels = self.count_occurrences(parent_labels,child_1["outcomes"])
        right_child_labels = self.count_occurrences(parent_labels,child_2["outcomes"])


        node['K'] = self.compute_k(left_probability,left_child_labels,
                                   right_probability,right_child_labels,parent_count)
        
        #Initial Filter to prevent overfitting
        if node['K'] < 15:
           return node["most_occuring_letter"]

    
        #Recursively call the function on the split dataset
        node["left"] = self.induce_decision_tree(child_1["attributes"], child_1["outcomes"])
        node["right"] = self.induce_decision_tree(child_2["attributes"], child_2["outcomes"])

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
        return data_set[index]

    def find_best_node_ideal(self, x, y):

        length, width = np.shape(x)
        class_labels,count = np.unique(y,return_counts=True)

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
                    #get entropys of each set
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


        #Returns the node
        return {"value": stored_value, "attribute": stored_attribute, "gain": stored_gain, "data": data, "left": None,
                "right": None,'K':None,'most_occuring_letter':None}

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

    def predict(self, x):

        # make sure that classifier has been trained before predicting
        if not self.is_trained:
            raise Exception("Decision Tree classifier has not yet been trained.")

        # set up empty N-dimensional vector to store predicted labels 
        # feel free to change this if needed
        predictions = np.zeros((x.shape[0],), dtype=np.object)
        
        
        # load the classifier
        tree = np.load('tree.npy',allow_pickle = True).item()
        
        for j in range(0, len(x)):
            predictions[j] = self.recursive_predict(tree, x[j][:])

        # remember to change this if you rename the variable
        return predictions

    def recursive_predict(self, tree, attributes):

        if not isinstance(tree,dict):
            return tree

        # Check the required attribute is greater or less than the node
        if attributes[tree["attribute"]] < tree["value"]:

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
            annotation = "IntNode:\n" + "\nAtt Split: "+ attributes[tree["attribute"]] 
            annotation += "<" + str(tree["value"]) 
            annotation +=  "\nIG:"+str(np.round(tree["gain"],3))
            center_x = mid_x
            center_y = y - height/2.0
            ax.annotate(annotation, (center_x,center_y), color='white',
                        weight='bold',fontsize=4, ha='center', va='center')
            plt.plot([parent_center_x,mid_x],[y+height,y],'black',linestyle=':',marker='')

        #Maximum depth to print out
        if depth ==3:
            return

        #else do all this stuff
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
        
        
    def cost_complexity_pruning(self,node):

        trees = np.array([],dtype=np.object)
        count = 0
        tree_copy = node.copy()
        
        while isinstance(tree_copy['left'],dict) or isinstance(tree_copy['right'],dict):
            trees = np.append(trees,self.prune_tree(tree_copy))
            count = count + 1
            tree_copy = tree_copy.copy()
            
        return trees


    def prune_tree(self,node):

        if not isinstance(node['left'],dict) and not isinstance(node['right'],dict):
            return node['most_occuring_letter']
        elif isinstance(node['left'],dict):
            node['left'] = self.prune_tree(node['left'])
            return node
        elif isinstance(node['right'], dict):
            node['right'] = self.prune_tree(node['right'])
            return node

        return node
    
    def calculate_best_pruned_tree(self,trees,x_test,y_test):
        
        eval = Evaluator()
        
        #go through each tree and compute the ratio of caculated error (right/total)
        for j in range(len(trees)):
            
            tree = trees[j]
            
            
    def get_wrong_prediciton_count(predictions,y_test):
        
        count = 0
        for j in range(len(predictions)):
            
            if predictions[j] != y_test[j]:
                count+=1
        
        return count
            
    


    """
    def Calculate_Classification_Loss(self,trees,evaluation_class,y_test):

        #while trees count is an instance of
        length = len(trees)
        classification_loss = np.zeros(length)

        while isinstance(trees[count],dict):

            #calculate the miss classfication rate


    def tree_test(self,tree):
        tree['right'] = 'A'

    """
