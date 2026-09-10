import copy
import unittest
import numpy as np
from contextual_feedback_study import shuffled_feedback

class ResumeTests(unittest.TestCase):
    def test_missing_or_partial_admission_result_cannot_be_shuffled(self):
        for original in [{'result':None,'outcome':'technical_error'}, {'result':{'trace':[]}}, {'result':{'trace':[{'contextual_projected':[0.]*32}]}}]:
            self.assertEqual(shuffled_feedback(original,2),(None,None))

    def test_shuffle_is_repeatable_and_preserves_input_and_width(self):
        original={'result':{'trace':[{'contextual_projected':[float(i)]*32} for i in range(8)]}}
        before=copy.deepcopy(original)
        a,permutation=shuffled_feedback(original,8);b,other=shuffled_feedback(original,8)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(permutation,other)
        self.assertEqual(a.shape,(8,32));self.assertEqual(original,before)
        self.assertEqual(sorted(permutation.tolist()),list(range(8)))

if __name__=='__main__':unittest.main()
