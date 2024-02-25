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
        if not std:
            if x == mu:
                return 1.0
            else:
                return 0.0
        expo_val = -0.5 * ((x - mu) / std) ** 2
        coeff_val = 1 / (std * np.sqrt(2 * np.pi))
        return coeff_val * np.exp(expo_val)
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
                feature_given_label = feature[(y == label)]
                self.params[feature_idx].append({'mean':np.mean(feature_given_label), 'std': np.std(feature_given_label)})
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
                prior = np.mean(self.y_train == label)
                ###################################

                # Calculating the posterior probability of the label
                log_likelihood = 0.0
                for i in range(len(x)):
                    ##############################
                    # YOUR CODE HERE #
                    ##############################
                    log_likelihood += np.log(1e-12+self._get_gaussian_likelihood(x[i], self.params[i][label]['mean'], self.params[i][label]['std']))
                prob_y_given_X[label] =  np.exp(log_likelihood) * prior
                #############################
            preds.append(np.argmax(prob_y_given_X))
        return np.array(preds)
