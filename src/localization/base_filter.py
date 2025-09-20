from abc import ABC, abstractmethod
import numpy as np
import cv2
from typing import Tuple

class BaseFilter(ABC):
    def __init__(self, map_array: np.ndarray):
        self.map = map_array
        self.HEIGHT, self.WIDTH = map_array.shape

    @abstractmethod
    def predict(self, fwd: float, turn: float) -> None:
        pass

    @abstractmethod
    def update(self, sensor_value: float) -> None:
        pass

    @abstractmethod
    def estimate_state(self) -> Tuple[float, float, float]:
        pass

    @abstractmethod
    def display(self) -> None:
        pass