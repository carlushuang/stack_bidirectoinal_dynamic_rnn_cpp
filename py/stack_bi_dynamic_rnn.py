import tensorflow as tf
import numpy as np
import argparse
import re

NUM_LAYERS  = 5
STEP_LENGTH = 3
HIDDEN_SIZE = 5
INPUT_SIZE  = 8
BATCH_SIZE = 1
FORGET_BIAS = 1.0

def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def zip_list_tuple(l1, l2):
    rtn = ()
    for i, j in zip(l1, l2):
        rtn = rtn + (i, j)
    return rtn

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step_length', type=int, default=STEP_LENGTH)
    parser.add_argument('--hidden_size', type=int, default=HIDDEN_SIZE)
    parser.add_argument('--input_size',  type=int, default=INPUT_SIZE)

    parser.add_argument('--batch_size',  type=int, default=BATCH_SIZE)
    parser.add_argument('--gen_data',    type=int, default=0)
    return parser.parse_args()

def main(args):
    step_length  = args.step_length
    hidden_size  = args.hidden_size
    input_size   = args.input_size
    batch_size   = args.batch_size
    gen_data     = args.gen_data
    num_layers = NUM_LAYERS

    if gen_data!=0 :
        print("... gen_data")
        # enter a cpu device scope
        with tf.device('/cpu:0'):
            # time_major == False` (default), this must be a `Tensor` of shape:, 
            #    `[batch_size, max_time, ...]`, or a nested tuple of such
            input_data = np.random.uniform( -1, 1, (batch_size, step_length, input_size)).astype(np.float32)
            input_data_ph = tf.placeholder( tf.float32, (batch_size, step_length, input_size))
            #init_state = (np.random.uniform(-1, 1, (batch_size, hidden_size)), np.random.uniform(-1, 1, (batch_size, hidden_size)))
            #init_state_ph = (tf.placeholder(tf.float32, (batch_size,hidden_size)), tf.placeholder(tf.float32, (batch_size,hidden_size)))
            init_state_fw_c = [np.random.uniform( -1, 1, (batch_size, hidden_size)).astype(np.float32) for _ in range(num_layers)]
            init_state_fw_h = [np.random.uniform( -1, 1, (batch_size, hidden_size)).astype(np.float32) for _ in range(num_layers)]
            init_state_fw_c_ph = [tf.placeholder( tf.float32, (batch_size, hidden_size)) for _ in range(num_layers)]
            init_state_fw_h_ph = [tf.placeholder( tf.float32, (batch_size, hidden_size)) for _ in range(num_layers)]
            init_state_fw = [tf.nn.rnn_cell.LSTMStateTuple(init_state_fw_c_ph[i], init_state_fw_h_ph[i]) for i in range(num_layers)]

            init_state_bw_c = [np.random.uniform( -1, 1, (batch_size, hidden_size)).astype(np.float32) for _ in range(num_layers)]
            init_state_bw_h = [np.random.uniform( -1, 1, (batch_size, hidden_size)).astype(np.float32) for _ in range(num_layers)]
            init_state_bw_c_ph = [tf.placeholder( tf.float32, (batch_size, hidden_size)) for _ in range(num_layers)]
            init_state_bw_h_ph = [tf.placeholder( tf.float32, (batch_size, hidden_size)) for _ in range(num_layers)]
            init_state_bw = [tf.nn.rnn_cell.LSTMStateTuple(init_state_bw_c_ph[i], init_state_bw_h_ph[i]) for i in range(num_layers)]

            weight_data_fw = [np.random.uniform(-1, 1, (input_size+hidden_size , 4*hidden_size)).astype(np.float32)] + \
                                [np.random.uniform(-1, 1, (hidden_size*2+hidden_size , 4*hidden_size)).astype(np.float32) for _ in range(num_layers-1)]
            bias_data_fw = [np.random.uniform(-1, 1, (4*hidden_size)).astype(np.float32) for _ in range(num_layers)]
            weight_data_bw = [np.random.uniform(-1, 1, (input_size+hidden_size , 4*hidden_size)).astype(np.float32)] + \
                                [np.random.uniform(-1, 1, (hidden_size*2+hidden_size , 4*hidden_size)).astype(np.float32) for _ in range(num_layers-1)]
            bias_data_bw = [np.random.uniform(-1, 1, (4*hidden_size)).astype(np.float32) for _ in range(num_layers)]

            #lstm_cell = tf.contrib.rnn.BasicLSTMCell ( hidden_size, forget_bias=FORGET_BIAS, state_is_tuple=True)

            cells_fw = [tf.contrib.rnn.BasicLSTMCell ( hidden_size, forget_bias=FORGET_BIAS, state_is_tuple=True) for _ in range(num_layers)]
            cells_bw = [tf.contrib.rnn.BasicLSTMCell ( hidden_size, forget_bias=FORGET_BIAS, state_is_tuple=True) for _ in range(num_layers)]
 
            #the_run = tf.nn.dynamic_rnn ( lstm_cell, input_data_ph, initial_state=init_state, dtype=tf.float32)
            the_run = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw, cells_bw, input_data_ph,
                                        initial_states_fw=init_state_fw,  initial_states_bw=init_state_bw, dtype=tf.float32)
            #print("++++++++")
            #print(input_data_ph)
            #print(stack_lstm_layer)
            #print("++++++++")

            sess = tf.Session()
            with sess.as_default():
                sess.run( tf.global_variables_initializer() )
                #print("++++++++")
                for var in tf.trainable_variables():
                    if 'basic_lstm_cell' in var.name:
                        cell_index_str = re.search(r'cell_(\d+)', var.name).group(1)
                        cell_index = int(cell_index_str)
                        if '/fw/basic_lstm_cell/' in var.name:
                            #print("fw, ", cell_index, ", ", var.name)
                            if 'kernel' in var.name:
                                sess.run(tf.assign(var, weight_data_fw[cell_index]))
                            elif 'bias' in var.name:
                                sess.run(tf.assign(var, bias_data_fw[cell_index]))
                        elif '/bw/basic_lstm_cell/' in var.name:
                            #print("bw, ", cell_index, ", ", var.name)
                            if 'kernel' in var.name:
                                sess.run(tf.assign(var, weight_data_bw[cell_index]))
                            elif 'bias' in var.name:
                                sess.run(tf.assign(var, bias_data_bw[cell_index]))
                        #print(var)
                        #print("xxx ", cell_index)
                    #if 'kernel' in var.name:
                    #    sess.run(tf.assign(var, weight_data))
                    #elif 'bias' in var.name:
                    #    sess.run(tf.assign(var, bias_data))
                    #else:
                    #    pass
                _dict = merge_dicts({input_data_ph:input_data}, 
                                    {p:i for p, i in zip(init_state_fw_c_ph, init_state_fw_c)},
                                    {p:i for p, i in zip(init_state_fw_h_ph, init_state_fw_h)},
                                    {p:i for p, i in zip(init_state_bw_c_ph, init_state_bw_c)},
                                    {p:i for p, i in zip(init_state_bw_h_ph, init_state_bw_h)})

                output, out_state_fw, out_state_bw = sess.run(the_run, feed_dict=_dict)
                
                #print(output)
                #print(out_states)

                np.save("blob/input.npy", input_data)

                np.save("blob/init_state_fw.npy", np.concatenate(zip_list_tuple(init_state_fw_c, init_state_fw_h), axis=0))
                weight_tuple = ()
                for w, b in zip (weight_data_fw, bias_data_fw):
                    weight_tuple = weight_tuple + (w, np.expand_dims(b, axis=0))
                np.save("blob/weight_fw.npy", np.concatenate(weight_tuple , axis=0) )

                out_st = [[]]
                i=0
                for out_c, out_h in out_state_fw:
                    if i == 0:
                        out_st=np.concatenate((out_c, out_h ), axis=0)
                    else:
                        out_st=np.concatenate((out_st, out_c, out_h ), axis=0)
                    i=i+1
                #print(out_st)
                np.save("blob/output_state_fw.npy", out_st)

                np.save("blob/init_state_bw.npy", np.concatenate(zip_list_tuple(init_state_bw_c, init_state_bw_h), axis=0))
                weight_tuple = ()
                for w, b in zip (weight_data_bw, bias_data_bw):
                    weight_tuple = weight_tuple + (w, np.expand_dims(b, axis=0))
                np.save("blob/weight_bw.npy", np.concatenate(weight_tuple , axis=0) )
                out_st = [[]]
                i=0
                for out_c, out_h in out_state_bw:
                    if i == 0:
                        out_st=np.concatenate((out_c, out_h ), axis=0)
                    else:
                        out_st=np.concatenate((out_st, out_c, out_h ), axis=0)
                    i=i+1
                #print(out_st)
                np.save("blob/output_state_bw.npy", out_st)

                np.save("blob/output.npy", output)
                #np.save("blob/lstm_weight.npy", np.concatenate((weight_data, np.expand_dims(bias_data, axis=0)) , axis=0) )
                #out_c, out_h = out_states
                #np.save("blob/lstm_output_state.npy", np.concatenate((out_c, out_h), axis=0) )
                #np.save("blob/lstm_output.npy", output)
                #print("+++++++")
                #print(out_c)
                #np.save("blob/lstm_output_state.npy", np.concatenate((out_c, out_h), axis=0))
                #print(output)
                #np.save("blob/lstm_output.npy", output.eval())
                #np.save("blob/ls")
                #for var in tf.trainable_variables():
                #    if 'kernel' in var.name:
                #        print(var)
                #        print(var.eval())
                #    elif 'bias' in var.name:
                #        print(var)
                #        print(var.eval())
    else:
        print("... run")


if '__main__' == __name__ :
    args = init_args()
    main(args)
