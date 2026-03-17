from enum import Enum

class QuantType(Enum):
    Q16Bit = 0
    Q8Bit = 1
    Q4Bit = 2

    
class QuantBackend(Enum):
    NONE = 0
    QUANTO = 1
    BITSANDBYTES = 2


class QuantTarget(Enum):
    TEXT_ENCODER = 0
    TRANSFORMER = 1