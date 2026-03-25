from enum import Enum

class ProcessType(Enum):
    TextToImage = 0
    ImageToImage = 1
    ImageEdit = 2
    ImageInpaint = 3
    ImageControlNet = 4,
    ImageToImageControlNet = 5,

    TextToVideo = 300
    ImageToVideo = 301
    VideoToVideo = 302


class MemoryMode(Enum):
    OffloadGPU = 0
    OffloadCPU = 1
    OffloadModel = 2
    Balanced = 3


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