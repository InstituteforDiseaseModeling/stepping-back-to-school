__all__ = ['ReferenceTrajectory']

class ReferenceTrajectory():
    ''' Mostly a placeholder for now '''
    def __init__(self, targets):
        self.targets = targets

        self.dynamics = 'static'

    '''
    def error_dynamics(self):
        return some matrix
    '''

    def get(self, t):
        return self.targets['infected']
