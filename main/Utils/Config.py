from .ArgParser import ArgParser
import json

class Config:
    CONF_FILE_PATH = 'configuration.json'

    def __init__(self, argParserObj):
        if argParserObj is None or not isinstance(argParserObj, ArgParser):
            raise TypeError('argParserObj was passed to config uninitialized')

        self.argParserObj = argParserObj

        with open(CONF_FILE_PATH) as f:
            self.confFile = json.load(f)

    def getSetting(self, settingName):
        if hasattr(self.argParserObj, settingName):
            return getattr(self.argParserObj, settingName)
        else if settingName in self.confFile:
            return self.confFile[settingName]
        else:
            raise KeyError(
                'Setting %s not in args or configuration file' % settingName
            )

