import unittest

import numpy as np

from spn.algorithms.Inference import log_likelihood
from spn.algorithms.MPE import mpe
from spn.io.CPP import get_cpp_function, setup_cpp_bridge, get_cpp_mpe_function
from spn.io.Graphics import plot_spn
from spn.structure.Base import get_nodes_by_type 
from spn.structure.leaves.parametric.Inference import add_parametric_inference_support
from spn.structure.leaves.parametric.Parametric import Gaussian,Bernoulli

class TestCPP(unittest.TestCase):
    def setUp(self):
        add_parametric_inference_support()
        
    # def test_binary(self): 
    #     D = Bernoulli(p=0.2, scope=[0])
    #     E = Bernoulli(p=0.6, scope=[1])
    #     F = Bernoulli(p=0.4, scope=[0])
    #     G = Bernoulli(p=0.7, scope=[1])

    #     B = D * E
    #     C = F * G

    #     A = 0.3 * B + 0.7 * C
    #     import pickle
    #     with open('spn_small.bin', 'wb') as f:
    #         pickle.dump(A, f)
        
    #     spn_cc_eval_func_bernoulli = None
    #     try:
    #         spn_cc_eval_func_bernoulli = get_cpp_function(A)
    #     except:
    #         setup_cpp_bridge(A)
    #         spn_cc_eval_func_bernoulli = get_cpp_function(A)

    #     num_data = 200000
    #     # np.random.seed(15)
    #     data = np.random.binomial(1, 0.3, size=(num_data)).astype('float32').tolist() \
    #          + np.random.binomial(1, 0.3, size=(num_data)).astype('float32').tolist()
    #     data = np.array(data).reshape((-1, 2))

    #     num_nodes = len(get_nodes_by_type(A))

    #     # print(type(data))
    #     # print(data)
    #     lls_matrix = np.zeros( (num_data, num_nodes) )

    #     py_ll = log_likelihood(A, data, lls_matrix=lls_matrix)
    #     c_ll = spn_cc_eval_func_bernoulli(data)
    #     # self.assertEqual(py_ll.shape[0], c_ll.shape[0])
    #     for i in range(lls_matrix.shape[0]):
    #         print("pyl: {val}".format(val = py_ll[i]))
    #         print("pyy: {val}".format(val = lls_matrix[i, :]) )
    #         print("cpp: {val}\n".format(val = c_ll[i, :] ) )
    #         for j in range(lls_matrix.shape[1]):
    #             self.assertAlmostEqual(lls_matrix[i, j], c_ll[i, j])

    def test_binary_mpe(self):
        # D = Bernoulli(p=0.2, scope=[0])
        # E = Bernoulli(p=0.6, scope=[1])
        # F = Bernoulli(p=0.4, scope=[0])
        # G = Bernoulli(p=0.7, scope=[1])

        # B = D * E
        # C = F * G

        # A = 0.3 * B + 0.7 * C

        A = 0.4 * (Bernoulli(p=0.8, scope=0) *
                    (0.3 * (Bernoulli(p=0.7, scope=1) *
                            Bernoulli(p=0.6, scope=2))
                    + 0.7 * (Bernoulli(p=0.5, scope=1) *
                            Bernoulli(p=0.4, scope=2)))) \
            + 0.6 * (Bernoulli(p=0.8, scope=0) *
                    Bernoulli(p=0.7, scope=1) *
                    Bernoulli(p=0.6, scope=2))
        import pickle
        with open('spn_small.bin', 'wb') as f:
            pickle.dump(A, f)

        num_inputs = len(A.scope)
        setup_cpp_bridge(A)

        num_data = 20000
        # np.random.seed(15)

        evidence_data = []

        # need to use tolist and concatenate for memory to be contiguous. 
        for _ in range(num_inputs): 
            evidence_data += np.random.binomial(1, 0.3, size=(num_data)).astype('float32').tolist() 
        evidence_data = np.array(evidence_data).reshape((-1, num_inputs))

        for i in range(evidence_data.shape[0]):
            drop_data = np.random.binomial(evidence_data.shape[1] - 1, 0.5)
            evidence_data[i, drop_data] = np.nan

        from cppyy.gbl import spn_mpe_many
        cc_completion = np.zeros_like(evidence_data)
        spn_mpe_many(evidence_data, cc_completion, evidence_data.shape[1], evidence_data.shape[0])        
        # print( [evidence_data, cc_completion] )

        py_completion = mpe(A, evidence_data)
        # for i in range(num_data):
        #     print( evidence_data[i, :] )
        #     print( cc_completion[i, :] )
        #     print( py_completion[i, :] )
        #     print("\n")
        # print(cc_completion)
        # print(py_completion)
        self.assertTrue( np.allclose(py_completion, cc_completion) )


        
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

    #     num_data = 20
    #     np.random.seed(17)
    #     data = np.random.normal(10, 0.01, size=num_data).tolist() \
    #         + np.random.normal(30, 10, size=num_data).tolist()
    #     data = np.array(data).reshape((-1, 2))

    #     lls_matrix = np.zeros(shape=(data.shape[0], len(get_nodes_by_type(A) )), 
    #                             dtype='float32')

    #     py_ll = log_likelihood(A, data, lls_matrix = lls_matrix)

    #     c_ll = spn_cc_eval_func_gaussian(data)


    #     for i in range(lls_matrix.shape[0]):
    #         print("pyl: {val}".format(val = py_ll[i]))
    #         print("pyy: {val}".format(val = lls_matrix[i, :]) )
    #         print("cpp: {val}\n".format(val = c_ll[i, :] ) )

    #         for j in range(lls_matrix.shape[1]): 
    #             self.assertAlmostEqual(lls_matrix[i, j], c_ll[i, j], places=4)



if __name__ == "__main__":
    unittest.main()
