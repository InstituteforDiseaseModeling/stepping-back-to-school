import numpy as np

class Kalman():
    def __init__(self, initial_exposed_pop, SEIR):

        self.SEIR = SEIR
        self.nEI = self.SEIR.EI.shape[0]
        self.nIR = self.SEIR.IR.shape[0]

        # Filtered state estimate
        self.nEIR = self.nEI + self.nIR
        self.EIhat = np.matrix(np.zeros( (self.nEIR,1) )) # TODO: Better guess value than all zeros!
        self.EIhat[0,0] = initial_exposed_pop
        self.Sigma = 1 * np.eye(self.nEIR) # State estimate covariance # TODO: Better guess

        self.Q = 10 * np.eye(self.nEIR) # Process noise - TODO: better guess
        self.R = 1 * np.eye(2)  # Observation noise (E and I) # TODO


    def update(self, yE, yI, u):
        # remove S and R states, leaving just E and I
        nERR = self.SEIR.nERR
        A = self.SEIR.A[1:-1-nERR,1:-1-nERR]
        B = self.SEIR.B[1:-1-nERR]

        Ce = np.hstack([np.ones(self.nEI), np.zeros(self.nIR)])
        Ci = np.hstack([np.zeros(self.nEI), np.ones(self.nIR)])
        C = np.matrix(np.vstack([Ce, Ci]))
        y = np.vstack([yE, yI])

        print('BEFORE:', self.EIhat.T)
        #print(self.Sigma)

        # Predict
        self.EIhat = A * self.EIhat + B * u
        self.Sigma = A * self.Sigma * A.T + self.Q

        # Correct
        L = self.Sigma * C.T * np.linalg.inv(C*self.Sigma*C.T + self.R)
        self.EIhat += L * (y - C*self.EIhat)
        self.Sigma = (np.eye(self.nEIR) - L*C)*self.Sigma

        print('AFTER:', self.EIhat.T)
        print('Corrected error:', (y-C*self.EIhat).T)
        #print(self.Sigma)


    def Ehat(self):
        return self.EIhat[:self.nEI]

    def Ihat(self):
        return self.EIhat[self.nEI:]
