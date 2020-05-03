import argparse

class ArgParser:
    DESCRIPTION_STR = '''
        Trains the main (currently decagon) model. Code is modular, and provides
        a series of configuration options for modules to use (e.g., learning policy,
        optimizer, graph builder).
    '''

    def __init__(self):
        self.args = None

        self.parser = argparse.ArgumentParser(description=self._getDescription())
        self._addArgs()

    def __getattr__(self, key):
        if hasattr(self.args, key):
            return getattr(self.args, key)
        else:
            raise AttributeError(self._getErrStr(key))

    def _getErrStr(self, key):
        errStr = ''
        if self.args is None:
            errStr = 'Parse has not yet been called on ArgParser object'
        elif not hasattr(self.args, key):
            errStr = 'Did not have arg %s in parsed args' % key
        else:
            errStr = 'Arg %s was not specified in process invocation' % key

        return errStr

    def _addArgs(self):
        self.parser.add_argument()

    def parse(self):
        self.args = self.parser.parse_args()

