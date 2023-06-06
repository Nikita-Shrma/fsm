
import numpy as np


class LinearRegression:
    # fom here algorithm starts, it is like constructor in python.as a defalut parameer, self is passed as an argument is is like the object of the class
    # which is object variable used to initialize the data members ,
       # it has default parameter learning_rate=0.001, n_iters=1000,i.e. although I have passes learning rate, 
       # n_iters value as 0.06 ,1000 but if there is some problem so it will take k as 0.001 insead of 0.06  and start 
    # running the code according to value of learning rate as 0.001
   
    # y=wx+b  where w= weight and b= bias which is  
    def __init__(self, learning_rate=0.001, n_iters=1000): 
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

 #this is the function fit inorder to fit the values ; what needs to be features for calculation and what is the ans which will be used for learning
    #in this features like 'age', 'sex', 'bmi', 'children', 'smokerd etc are used as x_train which will be used fo calculation 
    #y_train have the values of column charges  person having particular features will be charged how much this information  is there in this fit function
    #in a nutshell, it is a training dataset that will help in governing answers for test dataset
    def fit(self, X, y):
        n_samples, n_features = X.shape  # n_samples is number of rows , n__features= no. of columns in dataset

        #now initialization of some other factors wih zero
        self.weights = np.zeros(n_features) #weights basically tells how much important is the features for calculation and determination of answer
        self.bias = 0 #bias is constant added to the final answer

       
        for _ in range(self.n_iters): # running the process n_iter times
            y_predicted = np.dot(X, self.weights) + self.bias # now using formula y=wx+b final ans is calculated
            # compute gradients
            #so formula for dw is 1/N* (2*x.(predicted-given value)) {"x." is dot product}
            #formula for db is mean of (2*(predicted-given value)) so 2 is constant hence ignored
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y)) 
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # now final parametes are w=w-lr*dw,b=b-lr*db
            # now lr is learning rate it represents how slowly we move towards descent at one go ; it is uasually taken low
            # this is done to get lowest mse possible 
            self.weights -= self.lr * dw 
            self.bias -= self.lr * db

    def predict(self, X): # in prediction we pass test data 
        y_approximated = np.dot(X, self.weights) + self.bias # this is simply prediction of charge value 
        return y_approximated




    