import unittest
from scipy import stats
from AB_testing import *


class AdTests(unittest.TestCase):
    def test_prior_update(self):
        # Uniform prior
        A = Product(1, 1, 10)
        # This should be strong evidence that the conversion rate is 10% so value is 100K
        A.update_beliefs(10000, 1000)
        self.assertAlmostEqual(A.expected_value, 100, msg="Should update prior correctly", delta=0.1)

    def test_zero_value(self):
        """
        In this case, we're very confident that B is better than A. In this case, the VOI should be essentially zero
        because it is practically impossible for a test result to change our mind.
        """
        A = Product(1000, 9000, 10)
        B = Product(9000, 1000, 10)
        voi = calc_voi(A, B, test_sample_size=1000)
        self.assertAlmostEqual(voi, 0.0, msg="The VOI of two confidently different distributions is zero", delta=1)

    def test_clairvoyance_A(self):
        """
        Testing that the value of clairvoyance is close to that of a large test when B is better in the prior
        """
        A = Product(5, 50, 10)
        B = Product(2, 50, 10)
        voi = calc_voi(A, B, test_sample_size=20000, num_iter=50000)
        voc = calc_voc(A, B)
        self.assertAlmostEqual(voi, voc, msg="The VOI of a very large test should be close to the VOC", delta=.1)

    def test_clairvoyance_B(self):
        """
        Testing that the value of clairvoyance is close to that of a large test when B is better in the prior
        """
        A = Product(16, 50, 10)
        B = Product(19, 50, 10)
        voi = calc_voi(A, B, test_sample_size=20000, num_iter=50000)
        voc = calc_voc(A, B)
        self.assertAlmostEqual(voi, voc, msg="The VOI of a very large test should be close to the VOC", delta=.1)


if __name__ == '__main__':
    unittest.main()
