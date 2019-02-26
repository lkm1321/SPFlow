from observations import mnist
import tensorflow as tf
from tensorflow.python import debug as tf_debug

import numpy as np
import spn.experiments.RandomSPNs.RAT_SPN as RAT_SPN
import spn.experiments.RandomSPNs.region_graph as region_graph

import spn.algorithms.Inference as inference
import spn.io.Graphics as graphics


def one_hot(vector):
    result = np.zeros((vector.size, vector.max() + 1))
    result[np.arange(vector.size), vector] = 1
    return result


def load_mnist():
    (train_im, train_lab), (test_im, test_lab) = mnist("data/mnist")
    train_im_mean = np.mean(train_im, 0)
    train_im_std = np.std(train_im, 0)
    std_eps = 1e-7
    train_im = (train_im - train_im_mean) / (train_im_std + std_eps)
    test_im = (test_im - train_im_mean) / (train_im_std + std_eps)

    # train_im /= 255.0
    # test_im /= 255.0
    return (train_im, train_lab), (test_im, test_lab)


def train_spn_em(spn, train_im, num_epochs=50, batch_size=100, sess=tf.Session(), step_size=1.0): 

    input_ph = tf.placeholder(tf.float32, [batch_size, train_im.shape[1]])
    lls_result = dict()
    spn_output = spn.forward(input_ph, obj_to_tensor=lls_result)
    spn_output_mean = tf.reduce_sum(spn_output, axis=0) / float(batch_size)
    em_jobs = spn.backward_count(input_ph, lls_results = lls_result, root_counts = train_im.shape[0], step_size=step_size)
    # print(em_jobs)
    check_op = tf.add_check_numerics_ops()
    batches_per_epoch = train_im.shape[0] // batch_size

    sess.run(tf.global_variables_initializer())

    for i in range(num_epochs): 
        total_loss = 0
        for j in range(batches_per_epoch): 
            im_batch = train_im[j * batch_size : (j+1) * batch_size, :]

            result, cur_output, _ = sess.run( [em_jobs, spn_output_mean, check_op], feed_dict = {input_ph: im_batch})
            print(cur_output)
            # print( np.exp(result) )
            # print(type(result))
    
            total_loss += -cur_output[0]
        # for weight in result: 
        #     print( np.exp(weight) ) 

        print(i, total_loss / batches_per_epoch)

if __name__ == "__main__":
    rg = region_graph.RegionGraph(range(28 * 28))
    # rg = region_graph.RegionGraph(range(18))
    for _ in range(0, 10):
        rg.random_split(2, 4)

    args = RAT_SPN.SpnArgs()
    args.normalized_sums = True
    args.num_sums = 4
    args.num_univ_distros = 2
    # args.num_gauss = 
    args.leaf = 'bernoulli'
    spn = RAT_SPN.RatSpn(1, region_graph=rg, name="obj-spn", args=args)
    print("num_params", spn.num_params())


    (train_im, train_labels), _ = load_mnist()
    train_im = np.array( train_im > 0, dtype='float32')
    ones_idx = [idx for idx, val in enumerate(train_labels) if val == 1]
    train_im = train_im[ones_idx, :]

    # # print(train_im.shape)
    # # print(train_labels.shape)
    # train_spn(spn, train_im, train_labels, num_epochs=3, sess=sess)
    # train_im = np.array( [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0] * 1000 \
    #                     # + [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0] * 500 \
    #                     + [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0] * 1000).reshape( (-1, 18) )
    # np.random.shuffle(train_im)

    sess = tf.Session()
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    # sess.run(tf.global_variables_initializer())
    # simple_spn = spn.get_simple_spn(sess)

    # graphics.plot_spn(simple_spn[0])

    train_spn_em(spn, train_im, num_epochs=100, batch_size=1000, sess=sess, step_size=0.01)

    # dummy_input = np.random.normal(0.0, 1.2, [10, 9])
    # dummy_input = train_im[:5]
    # input_ph = tf.placeholder(tf.float32, dummy_input.shape)
    # output_tensor = spn.forward(input_ph)
    # tf_output = sess.run(output_tensor, feed_dict={input_ph: dummy_input})

    # output_nodes = spn.get_simple_spn(sess)
    # simple_output = []
    # for node in output_nodes:
    #     simple_output.append(inference.log_likelihood(node, dummy_input)[:, 0])
    # # graphics.plot_spn2(output_nodes[0])
    # # graphics.plot_spn_to_svg(output_nodes[0])
    # simple_output = np.stack(simple_output, axis=-1)
    # print(tf_output, simple_output)
    # simple_output = softmax(simple_output, axis=1)
    # tf_output = softmax(tf_output, axis=1) + 1e-100
    # print(tf_output, simple_output)
    # relative_error = np.abs(simple_output / tf_output - 1)
    # print(np.average(relative_error))
