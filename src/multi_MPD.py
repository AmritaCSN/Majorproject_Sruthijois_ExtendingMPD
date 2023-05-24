import numpy as np
from sklearn.utils import resample
from sklearn.base import clone


class UncertaintyTrainer:
    def __init__(self, X_train, y_train, base_clf, num_bootstrap=100):
        """
        :param bootstrap: number of samples to bootstrap.
        :param X_train: The training data to use for your MPD detector
        :param y_train: A vector of labels for the training data
        :base_clf: any sklearn estimator
        :num_bootstrap : number of times you want to bootstrap

        """
        self.X_train = X_train
        self.y_train = y_train
        self.num_bootstrap = num_bootstrap
        self.bootstrap_clfs = []
        self.base_clf = base_clf
        self.bootstrap_oodds= []
        self.total_probs = []
        self.num_classes = len(set(y_train))
        self.fit()

    def fit(self):
        """        
        This function resamples the X and y data and trains self.num_bootstrap models on them.
        It does not return anything, it sets self.bootstrap_clfs

        """

        for _ in range(self.num_bootstrap):
            X, y = resample(self.X_train, self.y_train, replace=True)
            clf = clone(self.base_clf)
            clf.fit(X, y)
            self.bootstrap_clfs.append(clf)

    
    def get_mpd_score(self, X):
        """
        :param X: An array of features samples to measure the MPD on
        
        :return: array of mpd values for the input array
        """

        probs = []
        mpd = np.full((len(X)),100)
        most_likely_mean_probs = np.full((len(X)),0)
        
       
        for j,clf in enumerate(self.bootstrap_clfs):  
            probs_classes = clf.predict_proba(X)
            probs.append(probs_classes)

        self.total_probs = np.array(probs)
        mean_probs  = []
        mpd_i = 0
        for i in range(self.num_classes):
            probs_i = self.total_probs[:,:,i]

            U_i = (probs_i-1)**2
            U_i = U_i.sum(axis=0)
            U_i = np.sqrt(U_i / self.num_bootstrap)

#             mean_probs = np.mean(probs_i,axis=0)
#             most_likely_mean_probs = np.maximum(most_likely_mean_probs,mean_probs)
            mpd = np.minimum(mpd, U_i)        
            
        return mpd