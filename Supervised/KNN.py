import numpy as np
def euclidean_distance(x1, x2):
    distance = (np.sum((x1-x2)**2))**0.5
    return distance

class KNN:
    # fom here algorithm starts, it is like constructor in python.as a defalut parameer, self is passed as an argument is is like the object of the class
    # which is object variable used to initialize the data members ,
    # it has default parameter k=3,i.e. although I have passes k value as 5 but if there issomeproblem so it will take k as 3 insead of 5 and start r
    # running the code according to vlaue of k as 3 
    def __init__(self, k):
        self.k = k
    #this is the function fit inorder to fit the values ; what needs to be features for calculation and what is the ans which will be used for learning
    #in this features like age, sex, fare, cabin,embarked are used as x_train which will be used fo calculation 
    #y_train have the values of column survived as 0,1 i.e. person having particular features been part of x-train will get survived or not this information is there in this fit function
    #in a nutshell, it is a training dataset that will help in governing answers for test dataset
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

# in def predict we are calling for calculation initiation for prediction here values of X_test is sent i.e. the values like
#age, sex, fare, cabin,embarked are now calculated in order to chheck or predict whether the information or feature provided about
# a customer is survived or not .so here from complete dataset X_test; single- single datapoints regarding a person is sent to calculate_pointwise
#and answer returned 0,1 will be stored in prediction which is returned back 
    def predict(self, X_test):
        predictions = [self.calculate_pointwise(x) for x in X_test]
        return predictions
# the function calculate_pointwise is called for len(X_test)times and for each dataset there is calculation of euclidean distances wrt X_train 
    def calculate_pointwise(self, x):
        # compute the euclidean distance,here each datapoint in X_train is calculated,ie.e euclidean distance between X(receive from predict function)andx_train is calculated
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # here we will get index of k closest neighbours or datapoints
        k_index = np.argsort(distances)[:self.k]
    # now wrt to indexes of people/passenger their values in y_train is stored in k_nearest_class
        k_nearest_class = [self.y_train[i] for i in k_index]

        #here, this is done ,in order to classify which are most common clas from which the point is related to
        # after that it will find out the index of class with most counts and return it ; which is actually a class from which data set belong to according to identification /classification of knn
       
    
        unique_class, class_counts = np.unique(k_nearest_class, return_counts=True)
        most_common_index = np.argmax(class_counts)
        most_common_class = unique_class[most_common_index]
        return most_common_class