#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 0-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 0-Clause License for more details.

class Sensor:
    def __init__(self, black_level, saturation, ccm, camera_name):
        self.black_level = black_level
        self.saturation = saturation
        self.ccm = ccm
        self.camera_name = camera_name

    def to_dict(self):
        return {'camera_name':self.camera_name, 'black_level': self.black_level, 'saturation': self.saturation, 'ccm': self.ccm}
