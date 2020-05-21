from .ArgParser import ArgParser
import json

class Config:
    CONF_FILE_PATH = 'configuration.json'

    def __init__(self, argParserObj: ArgParser) -> None:
        if argParserObj is None or not isinstance(argParserObj, ArgParser):
            raise TypeError('argParserObj was passed to config uninitialized')

        self.argParserObj: ArgParser = argParserObj

        with open(self._getConfFilename()) as f:
            self.confJson: Dict[str, str] = json.load(f)

    def _getConfFilename(self) -> str:
        if hasattr(self.argParserObj.args, 'config'):
            return self.argParserObj.getKey('config')
        else:
            return Config.CONF_FILE_PATH

    def getSetting(self, settingName: str):
        if hasattr(self.argParserObj, settingName):
            return self.argParserObj.getKey(settingName)
        elif settingName in self.confJson:
            return self.confJson[settingName]
        else:
            raise KeyError(
                'Setting %s not in args or configuration file' % settingName
            )

