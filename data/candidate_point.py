from dataclasses import dataclass, field
import numpy as np


@dataclass
class CandidatePoint:
    # prev_weights_list: list = field(default_factory=list)
    next_weights_list: list = field(default_factory=list)
    coordinate: np.ndarray = field(default_factory=lambda: np.zeros((1, 2)))
