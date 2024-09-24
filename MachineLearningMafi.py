import numpy as np
import matplotlib.pyplot as plt

class LinearRegression :
    def __init__ (self, learning_rate = 0.001 , iterations = 1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.theta = None

    def fit (self,X,y):
        """
        find the parameter theta

        Args:
            X (np array): features matrix (m,n)
            y (np array): target vector (m,1)
        """
        m,n = X.shape
        self.theta = np.zeros((n+1,1))
        one_vector = np.ones((m,1))
        X = np.hstack((one_vector,X))
        y = y.reshape(-1,1)

        for _ in range (self.iterations):
            y_pred = np.dot(X,self.theta)
            d_cost = (1/n)*np.dot(X.T , y_pred - y)
            self.theta -= self.learning_rate*d_cost


    def predict(self,X):
        """Predict the target values using the training model

        Args:
            X (np array): features matrix

        Returns:
            (np array): predicted target vector
        """
        m = X.shape[0]
        one_vector = np.ones((m, 1))
        X_ = np.hstack((one_vector, X))
        return np.dot(X_, self.theta)

    def score (self , y_pred ,y_real):
        """Returns the score (by 100) of the model using mean square

        Args:
            y_pred (np array): predicted target using the model
            y_real (np array): real values of the targets in the dataset
        """
        score = 100 * (1 - np.mean((y_real - y_pred) ** 2))
        return score
    



class LogisticRegression : 
    def __init__ (self, learning_rate = 0.001 , iterations = 1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = None
        self.b = None

    
    def fit (self , X , y):
        """Logistic regression classifier

        Args:
            X (np array): features matrix (m x n)
            y (np array): target vector (m x 1)
        """
        m,n = X.shape
        self.w = np.zeros((n,1))
        self.b = 0
        y= y.reshape(-1,1)
        epsilon = 1e-10
        cost_list = []

        for i in range (self.iterations):
            z = np.dot(X,self.w) + self.b
            A = 1/(1+np.exp(-z))
            cost = -(1/m)*np.sum( y*np.log(A + epsilon) + (1-y)*np.log(1-A + epsilon))
            dW = (1/m)*np.dot(X.T , A -y)
            dB = (1/m)*np.sum(A - y)
            self.w -= self.learning_rate *dW
            self.b -= self.learning_rate *dB
            cost_list.append(cost)
        plt.plot(np.arange(self.iterations),cost_list)
        plt.xlabel('iterations')
        plt.ylabel('cost value')
        plt.title('COST FUNCTION OVER ITERATIONS')
        plt.show()


    def predict (self,X):
        """
        Predict the target values using the training model

        Args:
            X (np array): features matrix

        Returns:
            (np array): predicted target vector
            """
        z = np.dot (X,self.w)+self.b
        A = 1/(1+np.exp(-z))
        return np.array ((A > 0.5).astype(int))



    def accuracy (self , X_test, y_test):
        """
        return the accuracy of the model on the testing dataset

        Args:
            X_test (np array): features matrix (m x n)
            y_test (np array): target vector (m x 1)

        Return:
            accuracy (float) : accuracy in %
        """
        z = np.dot (X_test,self.w)+self.b
        A = 1/(1+np.exp(-z))
        A = np.array ((A > 0.5).astype(int))
        accuracy = (1 - np.sum(np.absolute(A - y_test))/y_test.shape[0])*100
        return accuracy
    


class SVM:
    def __init__(self, kernel='linear', learning_rate=0.001, lambda_reg=0.01, iteration=1000, gamma=0.1, degree=3):
        """
        Args:
            kernel (str): The kernel type to be used in the algorithm.
                          Options: 'linear', 'polynomial', 'rbf'.
            
            learning_rate (float): Step size for gradient descent.
            lambda_reg (float): Regularization parameter.
            iteration (int): Number of iterations for gradient descent.
            gamma (float): Parameter for the RBF kernel.
            degree (int): Degree of the polynomial kernel.
            """
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.iteration = iteration
        self.gamma = gamma
        self.degree = degree
        self.kernel = kernel
        self.alpha = None
        self.b = None
        

    def _linear_kernel(self, x1, x2):
        K = np.dot(x1, x2)
        return K

    def _polynomial_kernel(self, x1, x2):
        K = (1 + np.dot(x1, x2)) ** self.degree
        return K

    def _rbf_kernel(self, x1, x2):
        K = np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)
        return K

    def _compute_kernel(self, x1, x2):
        if self.kernel == 'linear':
            return self._linear_kernel(x1, x2)
        elif self.kernel == 'polynomial':
            return self._polynomial_kernel(x1, x2)
        elif self.kernel == 'rbf':
            return self._rbf_kernel(x1, x2)
        
        else:
            raise ValueError("Unknown kernel")

    def fit(self, X, y):
        m, n = X.shape
        y_label = np.where(y <= 0, -1, 1)
        self.alpha = np.zeros(m)
        self.b = 0

        
        for _ in range(self.iteration):
            for i in range(m):
                condition = y_label[i] * (np.sum(self.alpha * self.y * [self._compute_kernel(X[i], x) for x in X]) + self.b) >= 1
                if condition:
                    self.alpha[i] -= self.learning_rate * (2 * self.lambda_reg * self.alpha[i])
                else:
                    
                    self.alpha[i] -= self.learning_rate * (2 * self.lambda_reg * self.alpha[i] - 1)
                    self.b -= self.learning_rate * y_label[i]

    def predict(self, X):
        # Predict based on the kernel trick, i.e., computing the dot product in the high-dimensional space
        predictions = []
        for x in X:
            decision = np.sum(self.alpha * self.y * [self._compute_kernel(x, x_train) for x_train in self.X]) + self.b
            predictions.append(np.sign(decision))
        return np.array(predictions)
    
    def accuracy (self,y_real,y_pred):
        accuracy = (np.sum(y_real == y_pred) / len(y_real))*100
        return f'accuracy is {accuracy}%'


        


