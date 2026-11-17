"""Fixed-parameter features used only to rank historical proposals."""
from collections import OrderedDict
import numpy as np

class Probes:
    """One all-ones feature per node, for ordering only; failed nodes remain."""

    def __init__(self):
        self.cache = OrderedDict()
        self.bytes = 0

    def get(self, data, node):
        mid = node["id"]
        if mid in self.cache:
            self.cache.move_to_end(mid)
            return self.cache[mid]
        try:
            feature=getattr(data,'reference_feature_for_proposals',data.feature)
            value = feature(node, [1.0] * len(node["params"]))
            value = value / max(float(np.linalg.norm(value)), 1e-30)
        except (ValueError, ArithmeticError, np.linalg.LinAlgError):
            value = None
        self.cache[mid] = value
        self.bytes += 0 if value is None else value.nbytes
        while self.bytes > 96 * 1024**2:
            _, old = self.cache.popitem(last=False)
            self.bytes -= 0 if old is None else old.nbytes
        return value

