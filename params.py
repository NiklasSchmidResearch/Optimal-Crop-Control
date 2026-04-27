import numpy as np
import numpy.typing as npt
from dataclasses import dataclass

@dataclass
class StochasticParameters:
    rainfalls: npt.NDArray[np.float64]
    initial_moisture: float
    initial_ammonium: float
    initial_nitrate: float