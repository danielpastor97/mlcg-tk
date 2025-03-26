from mlcg.nn.prior import Harmonic
from .utils import compute_raw_angles
 
class HarmonicRawAngle(Harmonic):
    _order = 3

    def __init__(self, statistics, name: str = "Angles") -> None:
        super(HarmonicRawAngle, self).__init__(statistics, name=name)

    @staticmethod
    def neighbor_list(topology) -> dict:
        return Harmonic.neighbor_list(topology, HarmonicRawAngle.name)

    @staticmethod
    def compute_features(pos, mapping):
        return compute_raw_angles(pos, mapping)