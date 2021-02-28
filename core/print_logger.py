#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 0-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 0-Clause License for more details.

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
