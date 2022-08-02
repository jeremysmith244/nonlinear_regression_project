import numpy as np
from numpy.linalg import pinv

class NLSolver():
    '''
    Class which can implements Gauss-Newton and Levenberg–Marquardt non-linear regression

    Taken directly from here:
    https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm

    Note, I found the wikipedia article about Gauss-Newton less clear than this article,
    and entire assumption of this project is these two are quite similar, and can be 
    encapulsated in a single class with shared methods.

    Constructor:
    b_init: np.array - initial guess for starting parameters
    x: np.array - x values for which you have reference data
    y: np.array - y references values, used as target for fit
    func: function - fitting function, must have form func(x,b), where b is np.array of len(b_init)
    method: str - Gauss-Newton and Levenberg–Marquardt
    lambd: float - Levenberg–Marquardt includes a damping parameter, which needs an initial value
    '''
    
    def __init__(self,
                 b_init:np.array, 
                 x:np.array,
                 y:np.array,
                 func:any,
                 method:str='Gauss-Newton',
                 lambd_init: float=1e-2):

        self.b = b_init
        self.x = x
        self.y = y
        self.func = func
        self.y_pred = self.func(x,self.b)
        self.lambd = lambd_init

        assert method in ['Gauss-Newton','Levenberg–Marquardt']
        self.method = method

    def fit(self,
            step_init:float=1e-6, 
            max_iter:int=100,
            stop_tolerance:float=1e-10):
        '''
        Canonical fit method, which will run through non-linear fit, and update self.b (parameters of fit)

        Input:
        step_init:float - initial step size delta for first round of iteration
        max_iter: int - maximum number of iterations of fitting
        stop_tolerance: float - if absolute value of rmse - rmse_prev is < this tolerance, considered converged
        '''

        b0 = self.b.copy()
        self.b = self.b + step_init

        if self.method == 'Gauss-Newton':
            for i in range(max_iter):
                rmse_prev = self.compute_rmse(self.y, self.func, self.x, self.b)
                r = self.compute_residual(self.y, self.func, self.x, self.b)
                J = self.compute_jacobian(self.x, self.b, b0, self.func, self.y)
                JpI = self.compute_pseudoinverse(J)
                b0 = self.b.copy()
                self.b = self.b + JpI @ r
                rmse = self.compute_rmse(self.y, self.func, self.x, self.b)

                print('Gauss-Newton; Iteration: %s, RMSE: %s'%(i,rmse))
                if np.abs(rmse_prev - rmse) < stop_tolerance:
                    break
                elif np.isnan(self.b).sum() > 0:
                    print('Fit diverged, trying different starting values')
                    break

        elif self.method == 'Levenberg–Marquardt':
            for i in range(max_iter):
                rmse_prev = self.compute_rmse(self.y, self.func, self.x, self.b)
                r = self.compute_residual(self.y, self.func, self.x, self.b)
                J = self.compute_jacobian(self.x, self.b, b0, self.func, self.y)
                JpI = self.compute_lm_inverse(J, self.lambd)
                b0 = self.b.copy()
                self.b = self.b + JpI @ r
                rmse = self.compute_rmse(self.y, self.func, self.x, self.b)

                print('Levenberg–Marquardt; Iteration: %s, RMSE: %s'%(i,rmse))
                if np.abs(rmse_prev - rmse) < stop_tolerance:
                    break
                elif np.isnan(self.b).sum() > 0:
                    print('Fit diverged, trying different starting values')
                    break
                # This is a very simple approach to updating lambda, 
                # called delayed gratification according to wikipedia
                elif rmse < rmse_prev:
                    self.lambd *= 0.33
                elif rmse > rmse_prev:
                    self.lambd *= 2


    def compute_jacobian(self, x:np.array, b:np.array, 
                         b0:np.array, func:any, y_ref:np.array):
        '''
        Numerically compute Jacobian with respect to b, at points x

        I found this very helpful when writing this:
        https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/jacobian/

        Input:
        x: np.array - points at which to evluate jacobian
        b: np.array - parameters to take jacobian wrt
        b0: np.array - previous parameters, informs size of local region for linearized approximation
        func: function - used to evaluate y at each point, in order to calculate residual
        y_ref: np.array - ground truth values, for computing residuals 

        Return:
        J: np.array - Jacobian wrt b, in local region b0-b at points x
        '''
        b = b.astype(float)
        b0 = b0.astype(float)
        J = np.zeros((len(x),len(b)))
        y0 = self.compute_residual(y_ref, func, x, b0)
        for i in range(J.shape[1]):
            b_partial = b0.copy()
            b_partial[i] = b[i]
            delta_residual = y0 - self.compute_residual(y_ref, func, x, b_partial)
            J[:,i] = delta_residual / (b[i] - b0[i])
        return J

    def predict(self):
        self.y_pred = self.func(self.x, self.b)
        return self.y_pred

    @staticmethod
    def compute_residual(y_ref:np.array, func:any, x:np.array, b:np.array):
        '''
        Compute residual based on reference, function, points and parameters

        Input:
        y_ref: np.array - ground truth values, for computing residuals 
        func: function - used to evaluate y at each point
        x: np.array - points at which to evluate residuals
        b: np.array - parameters for func
        '''
        return y_ref - func(x, b)

    @staticmethod
    def compute_rmse(y_ref:np.array, func:any, x:np.array, b:np.array):
        '''
        Compute rmse based on reference, function, points and parameters

        Input:
        y_ref: np.array - ground truth values, for computing residuals 
        func: function - used to evaluate y at each point
        x: np.array - points at which to evluate residuals
        b: np.array - parameters for func
        '''
        r = y_ref - func(x, b)
        return np.sqrt((r**2).mean())

    @staticmethod
    def compute_pseudoinverse(X:np.array):
        '''
        Compute Moore-Penrose pseuodinverse, other methods possible here (i.e., np.linalg.solve)

        Inputs:
        X: np.array - Input matrix to invert
        '''
        return pinv(X.T @ X) @ X.T

    @staticmethod
    def compute_lm_inverse(X, lambd):
        '''
        Compute inverse matrix for Levenberg–Marquardt, same as Gauss-Newton with added damping term

        Inputs:
        X: np.array - Input matrix to invert
        lambd: float - Damping factor for LM method
        '''
        return pinv(X.T @ X + lambd*np.diag(X.T @ X)) @ X.T

