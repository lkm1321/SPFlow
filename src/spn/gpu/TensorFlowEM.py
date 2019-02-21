"""
Created on March 27, 2018

@author: Alejandro Molina
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
from typing import Union, Tuple, List

from spn.algorithms.TransformStructure import Copy
from spn.structure.Base import Product, Sum, eval_spn_bottom_up, Node
from spn.structure.leaves.histogram.Histograms import Histogram
from spn.structure.leaves.histogram.Inference import histogram_likelihood
from spn.structure.leaves.parametric.Parametric import Gaussian
from spn.structure.Base import get_nodes_by_type
from spn.gpu.TensorFlow import spn_to_tf_graph
import logging

logger = logging.getLogger(__name__)

def leaf_gradient_backward_tf(
    node, parent_result, gradient_result=None, lls_matrix_tf=None
    ):
    gradient_result[:, node.id] = parent_result.reshape( (-1, 1) )

def sum_gradient_backward_tf(
    node, parent_result, gradient_result=None, lls_matrix_tf=None
):
    gradients = parent_result.reshape( (-1, 1) )

    gradient_result[:, node.id] = gradients
    num_child = len(node.children)

    messages_to_children = tf.zeros( shape=(gradients.shape[0], num_child) )

    with tf.variable_scope("{node_name}_{node_id}".format(
        node_name = node.__class__.__name__, node_id = node.id
        )):
        messages_to_children = gradients + tf.get_variable("weights")
    
    return messages_to_children

def prod_gradient_backward_tf(
    node, parent_result, gradient_result = None, lls_matrix_tf=None
): 
    gradient_result[:, node.id] = parent_result.reshape( (-1, 1) )

    messages_to_children = tf.zeros( shape())
    for c in node.children


def spn_em_to_tf_graph(spn, data, batch_size, dtype=tf.float32):
    
    nodes = get_nodes_by_type(spn)
    num_nodes = len(nodes)
    num_instances = data.shape[0]
    lls_matrix_tf = tf.zeros(shape=(num_instances, num_nodes) )

    tf_graph, data_placeholder, variable_dict = spn_to_tf_graph(
        spn, 
        data, 
        batch_size=batch_size, 
        dtype=dtype, 
        lls_matrix_tf=lls_matrix_tf
    )



