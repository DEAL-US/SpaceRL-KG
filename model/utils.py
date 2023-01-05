from data.data_manager import DataManager as dm

class Utils:
    """
    Utility functions class.

    :param verbose: if verbose is active.
    :param logs: if log generation is active.
    :param dm: the current data manager instance.
    """
    def __init__(self, verbose:bool, logs:bool, dm:dm = None):
        self.verbose = verbose
        self.logs = logs
        self.dm = dm

    def verb_print(self, msg:str):
        """
        Prints the requested message if verbose is active.

        :param str: the message to be printed.
        """
        if(self.verbose):
            print(msg)

    def write_log(self, msg:str):
        """
        Logs the requested message if logging is active

        :param str: the message to be logged.
        """
        if(self.logs):
            self.dm.write_log(msg)
