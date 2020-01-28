import numpy as np


from classification import DecisionTreeClassifier
from eval import Evaluator

if __name__ == "__main__":
    
    filename = "data/toy.txt"
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
                
    alphabet = np.array(["A","C","E","G","O","Q"])
    alphabet_count = np.zeros((len(alphabet)))
    alphabet_proportions = np.zeros((len(alphabet)))
    for indx,letter in enumerate(alphabet):
        for index in range(0,length):
            if(letter == y[index]):
                alphabet_count[indx]+=1
                
    alphabet_proportions[:] = np.round(alphabet_count[:]/length,3)
    
    print("Training the decision tree...")
    classifier = DecisionTreeClassifier()
    classifier = classifier.train(x,y)

    print("Loading the test set...")
    
    x_test = np.array([
            [1,6,3], 
            [0,5,5], 
            [1,5,0], 
            [2,4,2]
        ])
    #predictions = classifier.predict(x)
    predictions = classifier.predict(x_test)
    
    

