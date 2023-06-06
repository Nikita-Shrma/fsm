import numpy as np

class NaiveBayes:
# we start with function fit which has training fetures(X-train),training classes(y_train) in order to support for test data
    def fit(self, X_train, y_train):
        #N_samples is no. of rows, n_features in no. of coloumns in dataset train, we get this info from .shape
        n_samples, n_features = X_train.shape
        self._classes = np.unique(y_train)  # through this we get no. of unique classes in y_train(here we get  0,1 as answer)
        n_classes = len(self._classes) # through this we get length of classesi.e. number of uinque classes (ans in this will be 2 classes) 

        #now we have to calculate mean,variance and prior probability(prior probability is probability of class occuring in dataset i.e. probability
        # of 0 and probability of 1 in train dataset)
        # here is intialisation with zero
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64) # this initialized mean of  all the features wrt to classes as 0
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)  # this initialized var of  all the features wrt to classes as 0
        self._priors = np.zeros(n_classes, dtype=np.float64) # this initialized probability of classes as 0

        for idx, c in enumerate(self._classes): # here we calculate mean,variance,probability classwise   
            X_c = X_train[y_train == c]   # here we store features of class which have class ="c" from classes
            self._mean[idx, :] = X_c.mean(axis=0)  #here we calculate mean of all features at idx index 
            self._var[idx, :] = X_c.var(axis=0)   #similarly we calulate varianc
            self._priors[idx] = X_c.shape[0] / float(n_samples) # here probabilty of class 0,1 occuring in dataset is calculated 
            
 #here prediction of X_test starts.
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]  #here we call _predict function for calculation of probability
        return np.array(y_pred) 

    def _predict(self, x):
        posteriors = [] 

       #here we calculate probability 
       # posterior probability  is p(y)*p(x1|y)*p(x2|y)......p(xn|y) so all the values are b/w 0 &1 and multiplying them will make it very less
       # so we will take log and add all the factor i.e. log(p(y))+log(p(x1|y))+log(p(x2|y))........
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx]) #so here is log of p(y)(i.e.log of probabilty of class 0,1 occuring in dataset is calculated)
            posterior = np.sum(np.log(self._pdf(idx, x))) #here log(p(y))+log(p(x1|y))+log(p(x2|y)) is solved where p(x1|y) is calculated through_pdf function
            posterior = posterior + prior #here log(p(y)) and log(p(x1|y))+log(p(x2|y)) is added 
            posteriors.append(posterior) # value calculated above is added to list

        
        return self._classes[np.argmax(posteriors)] # class with highest probability is returned back.

# here we calculate gaussian function for p(x|y) 
# gaussian is  exp((x-mean)**2/((2*variance)))/sqroot(2*x*variance)
    def _pdf(self, class_idx, x): 
        mean = self._mean[class_idx] # here mean wrt to index is extracted out
        var = self._var[class_idx] # similarly variance
        numerator = np.exp(-((x - mean) ** 2) / (2 * var)) # here whole numerator is calculated
        denominator = np.sqrt(2 * np.pi * var) # here denominator is calculated 
        return numerator / denominator 



