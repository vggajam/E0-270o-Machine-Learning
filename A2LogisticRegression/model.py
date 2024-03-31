import numpy as np


class LinearModel:
    '''
    Linear model class.
    '''
    def __init__(self, inp_dim: int, out_dim: int = 1) -> None:
        '''
        Args:
            inp_dim: input dimension
            out_dim: output dimension
        
        Attributes:
            W: weight matrix
        '''
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.W = np.random.randn(inp_dim, out_dim) * (1 / np.sqrt(inp_dim))
        self.b = np.zeros(out_dim)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        '''
        Forward pass.

        Args:
            x: input data

        Returns:
            scalar output of linear model
        '''
        return x @ self.W + self.b

    def __call__(self, x: np.ndarray) -> np.ndarray:
        '''
        Forward pass.
        '''
        return self.forward(x)
    
    def __repr__(self) -> str:
        '''
        Representation of model.
        '''
        return f'LinearModel({self.inp_dim}, {self.out_dim})'


class LogisticRegression(LinearModel):
    def __init__(self, inp_dim: int) -> None:
        '''
        Args:
            inp_dim: input dimension
        '''
        super().__init__(inp_dim, 1)
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        '''
        Stable sigmoid function.

        Args:
            x: input data

        Returns:
            sigmoid of input data
        '''
        np_exp_x = np.exp(x)
        return np_exp_x/(1+np_exp_x)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        '''
        Forward pass.

        Args:
            x: input data

        Returns:
            scalar output of logistic regression model
        '''
        return self._sigmoid(super().forward(x)).reshape(-1)


class SoftmaxRegression(LinearModel):
    def __init__(self, inp_dim: int, out_dim: int = 10) -> None:
        '''
        Args:
            inp_dim: input dimension
            out_dim: output dimension
        '''
        super().__init__(inp_dim, out_dim)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        '''
        Stable softmax function.

        Args:
            x: input data

        Returns:
            softmax of input data
        '''
        np_exp_x = np.exp(x)
        np_exp_x_sum = np.sum(np_exp_x)
        return np_exp_x/np_exp_x_sum
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        '''
        Forward pass.

        Args:
            x: input data

        Returns:
            scalar output of softmax regression model
        '''
        return self._softmax(super().forward(x)).reshape(-1, self.out_dim)
