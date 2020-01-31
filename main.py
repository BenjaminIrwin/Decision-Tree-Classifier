import numpy as np


from classification import DecisionTreeClassifier
from eval import Evaluator

if __name__ == "__main__":
    
    filename = "data/toy.txt"
    
    print("Training the decision tree...")
    classifier = DecisionTreeClassifier()
    x,y = classifier.load_data(filename)
    classifier = classifier.train(x,y)

    print("Loading the test set...")
    
    x_test = np.array([
            [1,6,3], 
            [0,5,5], 
            [1,5,0], 
            [2,4,2]
        ])
    
    #filename = "data/test.txt"
    #x_test,y_test = classifier.load_data(filename)
    predictions = classifier.predict(x_test)
    print(predictions)
    
    classifier.print_tree()

    

        
            
    
