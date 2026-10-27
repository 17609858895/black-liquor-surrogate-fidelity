import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis/scripts"))
from twinc2st import sym_auc


class PairedScoreTests(unittest.TestCase):
    def test_excludes_self_comparisons(self):
        self.assertAlmostEqual(sym_auc(np.array([-2.0, 1.0, 1.0])), 1.0 / 3.0)

    def test_ties_receive_half_credit(self):
        self.assertAlmostEqual(sym_auc(np.array([-1.0, 1.0])), 0.5)


if __name__ == "__main__":
    unittest.main()
