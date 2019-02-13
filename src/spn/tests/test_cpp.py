import unittest

import numpy as np

from spn.algorithms.Inference import log_likelihood
from spn.algorithms.MPE import mpe
from spn.io.CPP import get_cpp_function, setup_cpp_bridge, get_cpp_mpe_function
from spn.structure.leaves.parametric.Inference import add_parametric_inference_support
from spn.structure.leaves.parametric.Parametric import Gaussian,Bernoulli

class TestCPP(unittest.TestCase):
    def setUp(self):
        add_parametric_inference_support()
        
    def test_binary(self): 
        D = Bernoulli(p=0.2, scope=[0])
        E = Bernoulli(p=0.6, scope=[1])
        F = Bernoulli(p=0.4, scope=[0])
        G = Bernoulli(p=0.7, scope=[1])

        B = D * E
        C = F * G

        A = 0.3 * B + 0.7 * C
        import pickle
        with open('spn_small.bin', 'wb') as f:
            pickle.dump(A, f)
        
        spn_cc_eval_func_bernoulli = None
        try:
            spn_cc_eval_func_bernoulli = get_cpp_function(A)
        except:
            setup_cpp_bridge(A)
            spn_cc_eval_func_bernoulli = get_cpp_function(A)

        np.random.seed(15)
        data = np.random.binomial(1, 0.3, size=(200000)).astype('float32').tolist() \
             + np.random.binomial(1, 0.3, size=(200000)).astype('float32').tolist()
        data = np.array(data).reshape((-1, 2))

        # print(type(data))
        # print(data)
        py_ll = log_likelihood(A, data)
        c_ll = spn_cc_eval_func_bernoulli(data)
        self.assertEqual(py_ll.shape[0], c_ll.shape[0])
        for i in range(py_ll.shape[0]):
            self.assertAlmostEqual(py_ll[i, 0], c_ll[i, 0])

    def test_binary_mpe(self):
        D = Bernoulli(p=0.2, scope=[0])
        E = Bernoulli(p=0.6, scope=[1])
        F = Bernoulli(p=0.4, scope=[0])
        G = Bernoulli(p=0.7, scope=[1])
        H = Bernoulli(p=0.2, scope=[2])
        I = Bernoulli(p=0.2, scope=[2])

        B = D * E * H
        C = F * G * I

        D2 = Bernoulli(p=0.2, scope=[3])
        E2 = Bernoulli(p=0.6, scope=[4])
        F2 = Bernoulli(p=0.4, scope=[3])
        G2 = Bernoulli(p=0.7, scope=[4])
        H2 = Bernoulli(p=0.2, scope=[5])
        I2 = Bernoulli(p=0.2, scope=[5])

        B2 = D2 * E2 * H2
        C2 = F2 * G2 * I2

        A1 = 0.3 * B + 0.7 * C 
        A2 = 0.7 * B2 + 0.3 * C2

        A = A1
        
        import pickle
        with open('spn_small.bin', 'wb') as f:
            pickle.dump(A, f)        

        try:
            spn_cc_mpe_func_bernoulli = get_cpp_mpe_function(A)
        except:
            setup_cpp_bridge(A)
            spn_cc_mpe_func_bernoulli = get_cpp_mpe_function(A)

        num_tests = 20
        np.random.seed(15)
        evidence_data = np.random.binomial(1, 0.3, size=(num_tests)).astype('float32').tolist() \
                      + np.random.binomial(1, 0.8, size=(num_tests)).astype('float32').tolist() \
                      + np.random.binomial(1, 0.7, size=(num_tests)).astype('float32').tolist() \
                      + np.random.binomial(1, 0.6, size=(num_tests)).astype('float32').tolist() \
                      + np.random.binomial(1, 0.5, size=(num_tests)).astype('float32').tolist() \
                      + np.random.binomial(1, 0.4, size=(num_tests)).astype('float32').tolist() 

        drop_data = np.random.binomial(1, 0.5, size=(num_tests, 6)).astype('bool') 

        evidence_data = np.array(evidence_data).reshape((-1, 6))
        evidence_data[drop_data] = np.nan
        # drop_idx = np.array( [0, 1] * (num_tests//2) ).reshape((-1, 1))
        # evidence_data[:, drop_idx] = np.nan
        # evidence_data[:, 0] = np.nan
        # evidence_data[evidence_data == 2] = 1
        
        # print(type(evidence_data[0, :]))
        # print(type(evidence_data[0, 0]))
        # print(evidence_data.shape)
        from cppyy.gbl import spn_mpe_many
        cc_completion = np.zeros_like(evidence_data)
        spn_mpe_many(evidence_data, cc_completion, evidence_data.shape[1], evidence_data.shape[0])        
        print( [evidence_data, cc_completion] )

        py_completion = mpe(A, evidence_data)
        # print(cc_completion)
        # print(py_completion)
        # self.assertTrue( np.allclose(py_completion, cc_completion) )


        
    # def test_bcpp(self):
    #     D = Gaussian(mean=20.0, stdev=5.0, scope=[0])
    #     E = Gaussian(mean=30.0, stdev=5.0, scope=[1])
    #     F = Gaussian(mean=40.0, stdev=5.0, scope=[0])
    #     G = Gaussian(mean=50.0, stdev=5.0, scope=[1])

    #     B = D * E
    #     C = F * G

    #     A = 0.3 * B + 0.7 * C

    #     spn_cc_eval_func_gaussian = None
    #     try:
    #         spn_cc_eval_func_gaussian = get_cpp_function(A)
    #     except:
    #         setup_cpp_bridge(A)
    #         spn_cc_eval_func_gaussian = get_cpp_function(A)

    #     np.random.seed(17)
    #     data = np.random.normal(10, 0.01, size=200000).tolist() + np.random.normal(30, 10, size=200000).tolist()
    #     data = np.array(data).reshape((-1, 2))
    #     print(type(data))

    #     print(data)
    #     py_ll = log_likelihood(A, data)

    #     c_ll = spn_cc_eval_func_gaussian(data)

    #     for i in range(py_ll.shape[0]):
    #         self.assertAlmostEqual(py_ll[i, 0], c_ll[i, 0])


if __name__ == "__main__":
    unittest.main()
