import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

def euclidean_distance(x1, x2): # here euclidean distance is calculated between centroid and sample; used for clustering
    return np.sqrt(np.sum((x1-x2)**2))

class KMeans:
 # fom here algorithm starts, it is like constructor in python.as a defalut parameer, self is passed as an argument is is like the object of the class
    # which is object variable used to initialize the data members ,
    # it has default parameter k=5,max_iters 100,i.e. although I have passes k value as 3,max_iters =150
    #  but if there is someproblem so it will take k as 5 insead of 3;ax_iters 100 instead of 150 and start 
    # running the code according to vlaue of k=5,max_iters 100
    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = K # here we store the data like k,max_iters
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        # for each cluster we put empty list for storing the index of smaple/row from dataset
        self.clusters = [[] for _ in range(self.K)]

        # we have empty list of centroids , to store centres for each cluster
        self.centroids = []


    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape #here n_samples is no. of rows and n_features=no. of cols in dataset

        # here we initialize the centroids, initially we take random points from samples 
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False) #here we get index ,here we get non-repetitive index as replace = False
        self.centroids = [self.X[idx] for idx in random_sample_idxs] # and wrt index centroid values is stored

        # here we do optimization of  clusters
        for _ in range(self.max_iters): # here we run the code for max_itertation time in order to cluster the data
            # here we assign datapoints/samples to closest centroids (in order to create cluster
            self.clusters = self._create_clusters(self.centroids) #so for clustering we call function _create_clusters

           

            # here we calculate new centroids from the new clusters 
            centroids_old = self.centroids # old centroid value get stored in centroids_old 
            self.centroids = self._get_centroids(self.clusters) # here self.centroids is updated using function _get_centroids from new clusters

            if self._is_converged(centroids_old, self.centroids): #however if we are getting same centroids each time we will stop the updation
                                                                  # we are checking old centroid and new centroid that was generated just above
                break     # if valuesare equal break out of function, it is done using func _is_converged

            

        # here we classify samples as the index of their clusters  through _get_cluster_labels
        return self._get_cluster_labels(self.clusters)


    def _get_cluster_labels(self, clusters):# here each sample from dataset will get the label of cluster wrt to assignment of data
        
        labels = np.empty(self.n_samples) # here we generate an empty array of n_samples size
        for cluster_idx, cluster in enumerate(clusters): #now owe have three cluster , and we will run one by one
            for sample_idx in cluster: #now in each cluster we have list with index , so we run the index
                labels[sample_idx] = cluster_idx #here list is filled wrt index

        return labels


    def _create_clusters(self, centroids):
        # here we cluster the data around centroids
        clusters = [[] for _ in range(self.K)]  # here we generate an empty array of k size i.e. for each cluster
        for idx, sample in enumerate(self.X): # here we get the index and sample 
            centroid_idx = self._closest_centroid(sample, centroids)# here we call another function _closest_centroid which will tell the distance between centroid and sample
            clusters[centroid_idx].append(idx) # now we store in clusters list wrt to index 
        return clusters

    def _closest_centroid(self, sample, centroids):
        # it will calculate distance of the current sample to each centroid using euclidean distance function
        distances = [euclidean_distance(sample, point) for point in centroids] #values are stored
        closest_idx = np.argmin(distances) #here we get index of min distance
        return closest_idx


    def _get_centroids(self, clusters):
        # here we calculate mean values of clusters and assign it as centroid
        centroids = np.zeros((self.K, self.n_features)) #here we initialize centroid with 0
        # now we calculate mean of each cluster
        for cluster_idx, cluster in enumerate(clusters): 
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean # here calculated value is assigned to centroid
        return centroids

    def _is_converged(self, centroids_old, centroids):
        # here we calculate distances between old and new centroids, for all centroids
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)] #loop run for k times
        return sum(distances) == 0  #if sum of distance is 0 we will return true 
    #above function is used to check whether the new centroid is equal to old , if condition is true then we will come out of loop

   