

class MatbenchError(Exception):
    """
    Exception specific to matbench methods.
    """

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return "AmsetError : " + self.msg