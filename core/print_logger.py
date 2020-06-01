# class to redirect all printed content to disk (but still print it)
class PrintLogger():
    def __init__(self, log_path, std):
        self._log = open(log_path, "a")
        self._terminal = std

    def write(self, message):
        self._terminal.write(message)
        self._log.write(message)

    def flush(self):
        self._log.flush()
        self._terminal.flush()

    def __del__(self):
        self._log.close()
