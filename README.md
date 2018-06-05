# stack_bidirectoinal_dynamic_rnn_cpp

a implmentation of tensorflow.contrib.rnn.stack_bidirectional_dynamic_rnn, with BasicLSTMCell, in c++

to test:
in "py" folder, install tensorflow (tested on tensorflow 1.8), run:
    python stack_bi_dynamic_rnn.py --gen_data=1
this will dump the input/output/state/weight into blob/*.npy

in "cpp" folder, run "build.sh", and run ./dynamic_rnn <path/to/py/blob>
you will see a lot output, and can check the result

** use https://github.com/rogersce/cnpy to read/write numpy ndarray
** tested all under ubuntu environment

