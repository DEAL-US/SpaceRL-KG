from data.data_manager import DataManager as dm

class Utils:
    def __init__(self, verbose, logs, dm:dm = None):
        self.verbose = verbose
        self.logs = logs
        self.dm = dm

    def verb_print(self, msg):
        if(self.verbose):
            print(msg)

    def write_log(self, msg):
        if(self.logs):
            self.dm.write_log(msg)
