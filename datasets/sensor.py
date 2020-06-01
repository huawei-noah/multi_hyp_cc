
class Sensor:
    def __init__(self, black_level, saturation, ccm, camera_name):
        self.black_level = black_level
        self.saturation = saturation
        self.ccm = ccm
        self.camera_name = camera_name

    def to_dict(self):
        return {'camera_name':self.camera_name, 'black_level': self.black_level, 'saturation': self.saturation, 'ccm': self.ccm}
