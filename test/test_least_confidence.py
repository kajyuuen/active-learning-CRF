import sys
sys.path.append('src')
import unittest
from least_confidence import calculate_least_confidences

class LeastConfidenceTest(unittest.TestCase):
    def setUp(self):
        self.predict_marginals = [
            [{'A': 1, 'B': 0}, {'A': 0, 'B': 1}],
            [{'A': 0.25, 'B': 0.75}, {'A': 0.75, 'B': 0.25}]
        ]

    def test_least_confidence(self):
        least_confidence = calculate_least_confidences(self.predict_marginals)
        self.assertEqual(least_confidence, [1, 0.75 * 0.75])

if __name__ == '__main__':
    unittest.main()
