from dataclasses import dataclass, field


@dataclass
class FilterResponse:
    p0: list = field(default_factory=list)
    p1: list = field(default_factory=list)
    H: float = 0.0
    first_id: int = 0
