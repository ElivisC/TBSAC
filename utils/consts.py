from enum import Enum


class Consts():
    Q_MAGENET = "QMAGNET"
    D_MAGENET_X = "DMAGNET_X"
    D_MAGENET_Y = "DMAGNET_Y"
    SOL = "SOL"


EnginesChoices = {
    'tracewin':'TraceWinEngine',
    'dnn':'DNNEngine',
    'accelerator':'AcceleratorEngine'
}

