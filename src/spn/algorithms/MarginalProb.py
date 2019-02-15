"""
Created on April 5, 2018

@author: Alejandro Molina
"""
import logging

import numpy as np

from spn.algorithms.Inference import log_likelihood
from spn.algorithms.Validity import is_valid
from spn.structure.Base import Leaf, Product, Sum, get_nodes_by_type, eval_spn_top_down
from spn.structure.leaves.parametric.Parametric import Bernoulli
from scipy.misc import logsumexp
import logging

logger = logging.getLogger(__name__)

def marginal_prob_prod(node, parent_result, data=None, lls_per_node=None, marginal_prob_log=None):
    # if len(parent_result) == 0:
    #     return None
    assert len(parent_result) == 1
    return [parent_result] * len(node.children)


def marginal_prob_sum(node, parent_result, data=None, lls_per_node=None, marginal_prob_log=None):
    # if len(parent_result) == 0:
    #     return None

    num_child = len(node.children)
    w_children_log_probs = np.zeros( (lls_per_node.shape[0], num_child ) )
    for i, c in enumerate(node.children):
        w_children_log_probs[:, i] = lls_per_node[:, c.id] + np.log(node.weights[i])

    # print(w_children_log_probs)
    # log( Prob(children selected) ) = log( Prob(Parent selected) ) + w_children_log_probs - logsumexp
    w_children_logsumexp = logsumexp(w_children_log_probs, axis=1)
    w_children_logsumexp = np.tile( logsumexp(w_children_log_probs, axis=1), (1, num_child) )

    # print(w_children_logsumexp)

    w_parent_log_prob = np.tile( parent_result, (1, num_child) )

    log_prob_children_selection = w_parent_log_prob + w_children_log_probs - w_children_logsumexp
    log_prob_children_selection = log_prob_children_selection.reshape(-1, 1)
    assert len(log_prob_children_selection) == num_child, "length is {length}, num_child is {num_child}".format(
        length=w_parent_log_prob, num_child=num_child
        )
    return log_prob_children_selection

def marginal_prob_leaf(node, parent_result, data=None, lls_per_node=None, marginal_prob_log=None):
    # if len(parent_result) == 0:
    #     return None

    # we need to find the cells where we need to replace nans with samples
    # data_nans = np.isnan(data[parent_result, node.scope])

    # data = observations by scope
    # data_in_scope = data[:, node.scope]
    # data_na

    # data_nans = np.array( [idx for idx in range(len(data)) if np.isnan( data[idx] ) ] )
    # n_samples = np.sum(data_nans)

    # if n_samples == 0:
    #     return None

    # if not isinstance(node, Bernoulli): 
    #     print("MarginalProb")

    # Doesn't really make much sense for other stuff than Bernoulli
    # print((marginal_prob_log[:, data_nans]).shape)
    # print(data_nans)
    # print(parent_result)

    # This leads to all ones (rightfully so, because it sums over all possible mixture variables)
    marginal_prob_log[:, node.id] = np.exp(parent_result)
    # marginal_prob_log[:, node.scope ] += np.tile( np.exp(parent_result), (1, len(node.scope) ) )
    # marginal_prob_log[:, node.scope ] += np.tile( np.log(node.p), (parent_result.shape[0], len(node.scope) ) )

_node_marginal_prob = {Product: marginal_prob_prod, Sum: marginal_prob_sum, Bernoulli: marginal_prob_leaf}

# def add_leaf_sampling(node_type, lambda_func):
#     _leaf_sampling[node_type] = lambda_func
#     _node_sampling[node_type] = marginal_prob_leaf

def add_node_sampling(node_type, lambda_func):
    _node_marginal_prob[node_type] = lambda_func

def get_marginal_prob(node, input_data, marginal_prob_funcs=_node_marginal_prob, log_space=False):
    """
    Get marginal probability at unseen input variables given input_data (i.e. the probability of a node being selected)
    """

    # first, we do a bottom-up pass to compute the likelihood taking into account marginals.
    # then we do a top-down pass, to sample taking into account the likelihoods.
    data = np.array(input_data)

    valid, err = is_valid(node)
    assert valid, err

    # assert np.all(
    #     np.any(np.isnan(data), axis=1)
    # ), "each row must have at least a nan value where the samples will be substituted"

    nodes = get_nodes_by_type(node)

    lls_per_node = np.zeros((data.shape[0], len(nodes)))

    log_likelihood(node, data, dtype=data.dtype, lls_matrix=lls_per_node)

    # Keep track of probability of leaf node selection. 
    leaf_node_prob = np.zeros(shape=(data.shape[0], len(nodes)), dtype='float32')
    marginal_prob_log = np.zeros(shape=data.shape, dtype='float32')
    # marginal_prob_log[ np.isnan(data) ] = 0.0

    eval_spn_top_down(
        node, 
        marginal_prob_funcs, 
        parent_result=np.zeros( shape=(data.shape[0], 1) ), 
        data=data, 
        lls_per_node=lls_per_node, 
        marginal_prob_log=leaf_node_prob
    )

    leaf_nodes = get_nodes_by_type(node, ntype=Bernoulli)
    leaf_node_ids = [leaf_node.id for leaf_node in leaf_nodes]
    leaf_node_scopes = [leaf_node.scope[0] for leaf_node in leaf_nodes]
    leaf_node_prob = leaf_node_prob[:, leaf_node_ids]
    print(leaf_node_prob)
    leaf_node_prob_true = [leaf_node.p * leaf_node_prob_val for (leaf_node, leaf_node_prob_val) in zip(leaf_nodes, leaf_node_prob) ]

    leaf_node_prob_true = np.array(leaf_node_prob_true)
    marginal_prob_log[:, leaf_node_scopes] = leaf_node_prob_true

    # print(data.shape)
    # bernoulli_prob = np.array( [np.log(leaf_node.p) for leaf_node in get_nodes_by_type(node, ntype=Bernoulli) ] )
    # print(bernoulli_prob.shape)
    # marginal_prob_log += 
    # marginal_prob = np.exp(marginal_prob_log)
    # print(np.exp(marginal_prob_log))
    print(marginal_prob_log)

    if log_space:
        return marginal_prob_log
    else:
        return marginal_prob_log