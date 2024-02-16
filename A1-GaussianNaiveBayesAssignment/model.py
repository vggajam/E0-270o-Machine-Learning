import numpy as np

class GaussianNaiveBayes:
    def __init__(self):
        self.params = []
        self.labels = []
        self.y_train = None

    def _get_gaussian_likelihood(
            self,
            x: float,
            mu: float,
            std: float,
    ) -> float:
        # Get Gaussian likelihood of x

        ####################################
        # YOUR CODE HERE
        ####################################
        # 
        #
        return None # Return the Likelihood
        ####################################

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray
    ) -> None:
        # Store the training labels
        self.labels = np.unique(y)
        self.y_train = y

        # Calculate the class conditional mean and std for each feature
        # (since we are assuming features are mutually independent)
        for feature_idx in range(X.shape[1]):
            self.params.append([]) # Should be populated with relevant parameters in the next loop
            feature = X[:, feature_idx]
            # Calculate the mean and std for each class for the given feature
            for label in self.labels:
                ###################################
                # YOUR CODE HERE
                ###################################
                #
                #
                ###################################

    def predict(
            self,
            X: np.ndarray
    ) -> np.ndarray:
        preds = []
        # Predict the label for each sample
        for x in X:
            prob_y_given_X = np.ones(len(self.labels)) # To be calculated
            for label in self.labels:
                # Calculating the prior probability of the label
                    
                ###################################
                # YOUR CODE HERE #
                ###################################
                #
                #
                ###################################

                # Calculating the posterior probability of the label
                for i in range(len(x)):
                    ##############################
                    # YOUR CODE HERE #
                    ##############################
                    #
                    #
                    #############################
            preds.append(np.argmax(prob_y_given_X))
        return np.array(preds)
