from NLSolver import *
import logging
import matplotlib.pyplot as plt
from time import time


logging.basicConfig(
    filename='./test_log.log',
    level=logging.INFO,
    filemode='w')

LOGGER = logging.getLogger()

def create_data():
    # Taken from wikipedia example for Gauss-Newton method

    S = np.array([0.038,0.194,0.425,0.626,1.253,2.5,3.74])
    r = np.array([0.05,0.127,0.094,0.2122,0.2729,0.2665,0.3317])

    def rate_eq(S,args):
        return args[0]*S / (args[1] + S)

    return S,r,rate_eq

def test_gauss_newton():

    LOGGER.info("Testing Gauss-Newton Method")
    S,r,rate_eq = create_data()
    B0 = np.array([0.1,0.1])

    t0 = time()
    try:
        gns = NLSolver(B0, S, r, rate_eq, method='Gauss-Newton')
        gns.fit()
        LOGGER.info("Gauss-Newton fitting: SUCCESS")
    except:
        LOGGER.error("Gauss-Newton fitting: FAILED")

    t1 = time()
    ypred = gns.predict()

    # Plot result
    plt.plot(S,r,'ro')
    plt.plot(S,ypred,'b--')
    plt.xlabel('Concentration')
    plt.ylabel('Rate')
    plt.savefig('Gauss-Newton Fit.png')
    plt.close()

    LOGGER.info('Testing Gauss-Newton in Python took: %s'%(t1-t0))

def test_levenberg_marquardt():

    LOGGER.info("Testing Levenberg–Marquardt Method")
    S,r,rate_eq = create_data()
    B0 = np.array([0.1,0.1])

    t0 = time()
    try:
        gns = NLSolver(B0, S, r, rate_eq, method='Levenberg–Marquardt')
        gns.fit()
        LOGGER.info("Levenberg–Marquardt fitting: SUCCESS")
    except:
        LOGGER.error("Levenberg–Marquardt fitting: FAILED")

    t1 = time()
    ypred = gns.predict()

    # Plot result
    plt.plot(S,r,'ro')
    plt.plot(S,ypred,'b--')
    plt.xlabel('Concentration')
    plt.ylabel('Rate')
    plt.savefig('Levenberg–Marquardt Fit.png')
    plt.close()

    LOGGER.info('Testing Levenberg–Marquardt in Python took: %s'%(t1-t0))