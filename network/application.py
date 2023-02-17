from enum import Enum


class ApplicationProfile(Enum):

    VIRTUAL_REALITY = (1.0, 100.0, 0.6)
    FACTORY_AUTOMATION = (2.0, 1.0, 0.8)
    DATA_BACKUP = (3.0, 2.0, 0.8)
    SMART_GRID = (4.0, 0.4, 0.03)
    SMART_HOME = (5.0, 0.001, 1.0)
    MEDICAL = (6.0, 0.2, 0.2)
    ENVIRONMENTAL_MONITORING = (7.0, 1.0, 0.1)
    TACTILE_INTERNET = (8.0, 120.0, 0.8)

    def __init__(self, id_, datarate, compression_factor):
        self.id = id_
        self.datarate = datarate
        self.compression_factor = compression_factor

    def __str__(self):
        return 'Application Profile datarate={}, compression_factor={}'.format(self.datarate, self.compression_factor)
