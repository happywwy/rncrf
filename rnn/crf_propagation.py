import numpy as np
from util.math_util import *
import random

# - QANTA's forward propagation. the labels argument indicates whether
#   you want to compute errors and deltas at each node or not. for training,
#   you obviously want those computations to occur, but during testing they
#   unnecessarily slow down feature computation

#define softmax function
def softmax(v):
    v = np.array(v)
    max_v = np.amax(v)
    e = np.exp(v - max_v)
    dist = e / np.sum(e)

    return dist

    
def der_tanh(x):
    return 1-np.tanh(x)**2

#def forward_prop(params, tree, d, labels=True):
def forward_prop(params, tree, d, c, labels=True):

    # node.finished = 0
    tree.reset_finished()

    to_do = tree.get_nodes()

    #(rel_dict, Wv, b, We) = params
    #(rel_dict, Wv, Wc, b, b_c, We) = params
    (rel_dict, Wv, b, We) = params

    # - wrong_ans is 100 randomly sampled wrong answers for the objective function
    # - only need wrong answers when computing error

    """
    if labels:
        random.shuffle(tree.ans_list)
        wrong_ans = [We[:, ind] for ind in tree.ans_list[0:100]]
    """

    # forward prop
    while to_do:
        curr = to_do.pop(0)

        # node is leaf
        if len(curr.kids) == 0:

            # activation function is the normalized tanh
            # compute hidden state
            curr.p = tanh(Wv.dot(curr.vec) + b)
            #curr.p_norm = curr.p / linalg.norm(curr.p)
            #curr.ans_error = 0.0
            #curr.label_error = 0.0
            #curr.label_delta = 0.0
            # wwy add classification
            #curr.predict_label = softmax(Wc.dot(curr.p) + b_c)

        else:

            # - root isn't a part of this! 
            # - more specifically, the stanford dep. parser creates a superficial ROOT node
            #   associated with the word "root" that we don't want to consider during training
            # 'root' is the last one to be popped
            if len(to_do) == 0:
                # 'root' only has one kid, which is the root word
                ind, rel = curr.kids[0]
                curr.p = tree.get(ind).p
                #curr.p_norm = tree.get(ind).p_norm
                #curr.ans_error = 0.
                #curr.label_error = 0.
                #curr.label_delta = 0.
                #curr.predict_label = softmax(Wc.dot(curr.p) + b_c)
                continue

            # check if all kids are finished
            all_done = True
            for ind, rel in curr.kids:
                if tree.get(ind).finished == 0:
                    all_done = False
                    break

            # if not, push the node back onto the queue
            if not all_done:
                to_do.append(curr)
                continue

            # otherwise, compute p at node
            else:
                kid_sum = zeros( (d, 1) )
                for ind, rel in curr.kids:
                    curr_kid = tree.get(ind)

                    try:
                        #kid_sum += rel_dict[rel].dot(curr_kid.p_norm)
                        kid_sum += rel_dict[rel].dot(curr_kid.p)

                    # - this shouldn't happen unless the parser spit out a seriously 
                    #   malformed tree
                    except KeyError:
                        print 'forward propagation error'
                        print tree.get_words()
                        print curr.word, rel, tree.get(ind).word
                
                kid_sum += Wv.dot(curr.vec)
                curr.p = tanh(kid_sum + b)
                #curr.p_norm = curr.p / linalg.norm(curr.p)
                
                #add prediction
                #curr.predict_label = softmax(Wc.dot(curr.p) + b_c)


        # error and delta
        if labels:
            """
            curr.ans_error = 0.0
            curr.label_error = 0.0
            curr.ans_delta = zeros( (d, 1) )
            curr.label_delta = zeros( (d, 1) )

            base = 1 - tree.ans_vec.T.dot(curr.p_norm)
            delta_base = -1 * tree.ans_vec.flatten()

            rank = 1.0
            for ans in wrong_ans:
                err = max(0.0, base + ans.T.dot(curr.p_norm))
                if err > 0.0:
                    # WARP approximation of rank
                    rank = (len(wrong_ans) - 1) / rank

                    # multiply curr.ans_error and curr.ans_delta by 1/rank for WARP effect 
                    curr.ans_error += err
                    delta = delta_base + ans
                    curr.ans_delta += delta.reshape( (d, 1))
                    rank = 0.0

                rank += 1
            """
            '''
            curr.label_error = 0.0
            curr.label_delta = zeros( (c, 1) )
            '''
            true_label = zeros( (c, 1) )
            for i in range(c):
                if curr.trueLabel == i:
                    true_label[i] = 1
                    
            curr.true_class = true_label
                    
            #curr.label_delta = curr.predict_label - curr.true_class
            #curr.label_error = - (np.multiply(log(curr.predict_label), curr.true_class).sum())

        curr.finished = 1


# computes gradients for the given tree and increments existing gradients
#def backprop(params, tree, d, len_voc, grads):
def backprop(params, tree, d, c, len_voc, grads, mixed = False):
    import numpy as np

    #(rel_dict, Wv, b) = params
    #(rel_dict, Wv, Wc, b, b_c) = params
    (rel_dict, Wv, b) = params

    # start with root's immediate kid (for same reason as forward prop)
    ind, rel = tree.get(0).kids[0]
    root = tree.get(ind)

    # operate on tuples of the form (node, parent delta)
    to_do = [ (root, zeros( (d, 1) ) ) ]

    while to_do:
        curr = to_do.pop()
        node = curr[0]
        #parent delta
        delta_down = curr[1]
        
        #delta_Wc
        #delta_Wc = node.label_delta.dot(node.p.T)    
        #delta_bc = node.label_delta
        
        #delta_node
        #delta = Wc.T.dot(node.label_delta)
        delta = node.grad_h
        curr_der = der_tanh(node.p)
        node.delta_node = np.multiply(delta, curr_der)
        
        node.delta_full = delta_down + node.delta_node

        # internal node
        if len(node.kids) > 0:
            

            #act = pd + node.ans_delta
            #node.delta_i = df.dot(act)

            for ind, rel in node.kids:

                curr_kid = tree.get(ind)
                #W_rel
                grads[0][rel] += node.delta_full.dot(curr_kid.p.T)
                #to_do.append( (curr_kid, rel_dict[rel].T.dot(node.delta_i) ) )
                to_do.append( (curr_kid, rel_dict[rel].T.dot(node.delta_full) ) )


            grads[1] += node.delta_full.dot(node.vec.T)
            #grads[2] += delta_Wc
            grads[2] += node.delta_full
            #grads[4] += delta_bc
            if mixed:
                #grads[5][50:, node.ind] += Wv.T.dot(node.delta_full).ravel()[50:]
                grads[3][50:, node.ind] += Wv.T.dot(node.delta_full).ravel()[50:]
            else:
                grads[3][:, node.ind] += Wv.T.dot(node.delta_full).ravel()

        # leaf
        else:
            #act = pd + node.ans_delta
            #df = dtanh(node.p)

            grads[1] += node.delta_full.dot(node.vec.T)
            #grads[2] += delta_Wc
            grads[2] += node.delta_full
            #grads[4] += delta_bc

            if mixed:
                grads[3][50:, node.ind] += Wv.T.dot(node.delta_full).ravel()[50:]
            else:
                grads[3][:, node.ind] += Wv.T.dot(node.delta_full).ravel()
