class Error(Exception):
    """
    This is the base class on which all local errors
    and exceptions are based.
    """
    pass

class FermiEnergyConflict(Error):
    """
    Exception to be raised when asking for a cost function which uses the fermi energy
    but not using finite the temperature which is used to calculate the fermi energy
    in the current implementation of the codebase.
    """
    def __init__(self, message):
        self.message = message

class ConvergenceFailure(Error):
    """
    Exception to be raised when encountering errors associated with
    convergence.

    Attributes:
        - message (str): An explanation of the error and why it occurred

    """

    def __init__(self, message):
        self.message = message


class DiagonalisationFailure(Error):
    """
    Exception to be raised when a matrix fails to diagonalize.
    """

    def __init__(self,message):
        self.message = message