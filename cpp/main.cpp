#include <iostream>
#include "cnpy.h"
#include <math.h>
#include <string>
#include <string.h>

static inline float sigmoid(float number) { return 1. / (1. + exp(-number)); }
static inline float tanh(float x){return (exp(2*x)-1)/(exp(2*x)+1);}

// tensorflow BasicLSTMCell implementation, single batch
// http://colah.github.io/posts/2015-08-Understanding-LSTMs/
//     (tensorflow/python/ops/rnn_cell_impl.py)
// forget_bias: float, The bias added to forget gates (see above).
//              Must set to `0.0` manually when restoring from CudnnLSTM-trained
// input        -> input_size x 1
// init_state   -> 2*hidden_size (cell, hidden)
// kernel       -> 4*hidden_size x input_size+hidden_size
// bias         -> 4*hidden_size x 1
// output       -> hidden_size (hidden_state is outputed)
// out_state    -> 2*hidden_size ( cell, hidden)
int basic_lstm_cell(float * input, float * init_state, float * kernel, float * bias,
    float forgat_bias, int input_size, int hidden_size,
    float * output, float * out_state){

    float * tmp = new float[4*hidden_size];

    int i, j;
    float * c_st = init_state;
    float * h_st = init_state + hidden_size;

    float * w_i = kernel;
    float * w_h = kernel + 4*hidden_size*input_size;

    // input * w_i -> tmp
    for(j=0;j<4*hidden_size;j++){
        float _v = 0;
        for(i=0;i<input_size;i++){
            _v += input[i] * w_i[i*4*hidden_size+j];
        }
        tmp[j] = _v;
    }

    // h_st * w_h ->tmp
    for(j=0;j<4*hidden_size;j++){
        float _v = 0;
        for(i=0;i<hidden_size;i++){
            _v += h_st[i] * w_h[i*4*hidden_size+j];
        }
        tmp[j] += _v;
        tmp[j] += bias[j];
    }

    // i = input_gate, j = new_input, f = forget_gate, o = output_gate
    float * t_i = tmp;
    float * t_j = tmp + hidden_size;
    float * t_f = tmp + hidden_size * 2;
    float * t_o = tmp + hidden_size * 3;

    float * new_c  = out_state;
    float * new_h  = out_state + hidden_size;

    //new_c = add(multiply(c, sigmoid(add(f, forget_bias_tensor))),
    //            multiply(sigmoid(i), self._activation(j)))
    //new_h = multiply(self._activation(new_c), sigmoid(o))

    for(i=0;i<hidden_size;i++){
        new_c[i] = c_st[i]*sigmoid(t_f[i]+forgat_bias) + sigmoid(t_i[i])*tanh(t_j[i]);
    }
    for(i=0;i<hidden_size;i++){
        new_h[i] = tanh(new_c[i])*sigmoid(t_o[i]);
    }
    for(i=0;i<hidden_size;i++){
        output[i] = new_h[i];
    }

    delete [] tmp;
    return 0;
}

int NUM_LAYERS  = 5;
int STEP_LENGTH = 3;
int HIDDEN_SIZE = 5;
int INPUT_SIZE  = 8;
int BATCH_SIZE = 1;
float FORGET_BIAS = 1.0;

static void dump_npy_array_info(const cnpy::NpyArray & arr, const std::string & m){
    std::cout<<m;
    std::cout<<"word_size:"<<arr.word_size<<", num_vals:"<<arr.num_vals;
    std::cout<<", shape:";
    for(auto & sp : arr.shape)
        std::cout<<sp<<",";
    std::cout<<std::endl;
}

typedef struct _rnn_type{

    int m_num_layers;
    int m_seq_length;
    int m_input_size;
    int m_num_units;        // hidden size
    bool m_bidirectional;
    float forget_bias ;

    float * ptr_weight_fw ; //(w,b) (w,b), ...  num layers
    //float * ptr_bias_fw   ;
    float * ptr_state_fw  ; // (cell_state, hidden_state)

    float * ptr_weight_bw ;
    //float * ptr_bias_bw   ;
    float * ptr_state_bw  ; // (cell_state, hidden_state)

}rnn_type;

/*
 * stack_bidirectional_dynamic_rnn(), single batch forward lstm
 * see tf.contrib.rnn.stack_bidirectional_dynamic_rnn()
 * here only care about LSTM cell
 *
 * input     ->  seq x input_size
 * output    ->  seq x hidden_size, or seq x 2*hidden_size for bidirection
 * out_state ->  2*hidden_size fw, 2*hidden_size bw(if bi)
 * 
 * require 2*hidden_size == input_size  (for bidirection)?
 */
int stack_bidirectional_dynamic_rnn(rnn_type * rnn, float * input, float * output, float * out_state_fw, float * out_state_bw)
{
    int i, j;
    float * input_cur_layer = input;
    int out_size = rnn->m_bidirectional ? 2*rnn->m_num_units:rnn->m_num_units;
    int input_size = rnn->m_input_size;
    //float * tmp_state = new float[2*rnn->m_num_units];
    //float * tmp_out = new float[rnn->m_seq_length * out_size];
    float * tmp_out_fw = new float [rnn->m_seq_length * rnn->m_num_units];
    float * tmp_out_bw = nullptr;
    if(rnn->m_bidirectional)
        tmp_out_bw = new float [rnn->m_seq_length * rnn->m_num_units];
    float * ptr_w_fw = rnn->ptr_weight_fw;
    float * ptr_w_bw = rnn->ptr_weight_bw;
    for(j=0;j<rnn->m_num_layers;j++){
        //if (j==1)
        //    break;
        int cur_kernel_num = 4*rnn->m_num_units * (input_size+rnn->m_num_units)/*kernel*/ ;
        int cur_bias_num = 4*rnn->m_num_units /*bias*/;
        int cur_weight_num = cur_kernel_num + cur_bias_num;
        for(i=0;i<rnn->m_seq_length;i++){
            float * cur_stat = rnn->ptr_state_fw + j*2*rnn->m_num_units;
            float * input_cur_seq = input_cur_layer +input_size*i;
            float * init_state = cur_stat;
            float * kernel = ptr_w_fw;
            float * bias = kernel +cur_kernel_num;
            float * output_cur_seq = tmp_out_fw+i*rnn->m_num_units;
            float * out_state = out_state_fw + j*2*rnn->m_num_units;

            basic_lstm_cell(input_cur_seq, init_state, kernel, bias,
                rnn->forget_bias,  input_size, rnn->m_num_units,
                output_cur_seq, out_state);

            // update cell state
            memcpy(cur_stat, out_state, 2*rnn->m_num_units*sizeof(float));
        }
        if(rnn->m_bidirectional){
            // reverse
            for(i=(rnn->m_seq_length-1);i>=0;i--){
                float * cur_stat = rnn->ptr_state_bw + j*2*rnn->m_num_units;
                float * input_cur_seq = input_cur_layer +input_size*i;
                float * init_state = cur_stat;
                float * kernel = ptr_w_bw;
                float * bias = kernel + cur_kernel_num;
                float * output_cur_seq = tmp_out_bw+i*rnn->m_num_units;
                float * out_state = out_state_bw + j*2*rnn->m_num_units;

                basic_lstm_cell(input_cur_seq, init_state, kernel, bias,
                    rnn->forget_bias,  input_size, rnn->m_num_units,
                    output_cur_seq, out_state);

                // update cell state
                memcpy(cur_stat, out_state, 2*rnn->m_num_units*sizeof(float));
            }
        }

        float * dest = output;
        if(rnn->m_bidirectional){
            // prev_layer = array_ops.concat(outputs, 2)
            int x,y;
            for(y=0;y<rnn->m_seq_length;y++){
                // fw
                for(x=0;x<rnn->m_num_units;x++){
                    *dest++ = tmp_out_fw[y*rnn->m_num_units+x];
                }
                // bw
                for(x=0;x<rnn->m_num_units;x++){
                    *dest++ = tmp_out_bw[y*rnn->m_num_units+x];
                }
            }
        }else{
            memcpy(output, tmp_out_fw, rnn->m_seq_length *rnn->m_num_units );
        }
        // refine input ptr and size
        input_size = out_size;
        input_cur_layer = output;
        // weight size may differenct in the 1st layer, so we have to do accumulate
        ptr_w_fw += cur_weight_num;
        ptr_w_bw += cur_weight_num;
        std::cout<<"__ out:";
        for(int ii=0;ii<out_size;ii++)
            std::cout<<output[ii]<<", ";
        std::cout<<std::endl;
    }
    delete [] tmp_out_fw;
    if(tmp_out_bw)
        delete [] tmp_out_bw;
    return 0;
}

// this is for single direction cells!!!
int cal_rnn_num_weights(int num_layers, int num_hidden, int num_input, bool bi){
    int inputs =num_input;
    int total = 0;
    while(num_layers--){
        total += 4*num_hidden * (inputs+num_hidden);
        inputs = bi ? 2*num_hidden:num_hidden;
    }
    return total;
}

int init_dynamic_rnn(rnn_type * rnn, int layers, int seq_len, int inputs, int hidden_size, bool bi,float forget_bias,
        float * init_st_fw, float * init_st_bw, float *w_fw, float * w_bw)
{
    rnn->m_num_layers = layers;
    rnn->m_seq_length = seq_len;
    rnn->m_input_size = inputs;
    rnn->m_num_units = hidden_size;
    rnn->m_bidirectional = bi;
    rnn->forget_bias = forget_bias;

    int num_weights = cal_rnn_num_weights(layers,hidden_size, inputs, bi);
    num_weights += layers * 4*hidden_size;
    int num_state = layers *2*hidden_size;

    rnn->ptr_weight_fw = new float[num_weights];
    rnn->ptr_state_fw  = new float[num_state];

#if 0
    std::cout<<"weight fw:"<<std::endl;
    for(int i=0;i<num_weights;i++){
        std::cout<<w_fw[i]<<", ";
    }
    
    std::cout<<std::endl;
    std::cout<<"weight bw:"<<std::endl;
    for(int i=0;i<num_weights;i++){
        std::cout<<w_bw[i]<<", ";
    }
    std::cout<<std::endl;
    std::cout<<"init fw:"<<std::endl;
    for(int j=0;j<layers;j++){
        for(int i=0;i<2*hidden_size;i++){
            std::cout<<init_st_fw[j*2*hidden_size + i]<<", ";
        }
        std::cout<<std::endl;
    }
    std::cout<<"init bw:"<<std::endl;
    for(int j=0;j<layers;j++){
        for(int i=0;i<2*hidden_size;i++){
            std::cout<<init_st_bw[j*2*hidden_size + i]<<", ";
        }
        std::cout<<std::endl;
    }
#endif

    if(w_fw)
        memcpy(rnn->ptr_weight_fw, w_fw, num_weights*sizeof(float));
    if(init_st_fw)
        memcpy(rnn->ptr_state_fw , init_st_fw, num_state*sizeof(float));
    else
        memset(rnn->ptr_state_fw, 0, num_state*sizeof(float));
    if(bi){
        rnn->ptr_weight_bw = new float[num_weights];
        rnn->ptr_state_bw = new float[num_state];
        if(w_bw)
            memcpy(rnn->ptr_weight_bw, w_bw, num_weights*sizeof(float));
        if(init_st_bw)
            memcpy(rnn->ptr_state_bw , init_st_bw, num_state*sizeof(float));
        else
            memset(rnn->ptr_state_bw, 0, num_state*sizeof(float));
    }else{
        rnn->ptr_weight_bw = nullptr;
        rnn->ptr_state_bw = nullptr;
    }
}
int destroy_dynamic_rnn(rnn_type * rnn){
    delete [] rnn->ptr_weight_fw;
    delete [] rnn->ptr_state_fw;
    if(rnn->m_bidirectional){
        delete [] rnn->ptr_weight_bw;
        delete [] rnn->ptr_state_bw;
    }
}


int main(int argc, char ** argv){
    if(argc<2)
        return -1;
    std::string blob_path = argv[1];
    cnpy::NpyArray arr_init_state_fw = cnpy::npy_load(blob_path+"/init_state_fw.npy");
    cnpy::NpyArray arr_init_state_bw = cnpy::npy_load(blob_path+"/init_state_bw.npy");
    cnpy::NpyArray arr_input = cnpy::npy_load(blob_path+"/input.npy");
    cnpy::NpyArray arr_weight_fw = cnpy::npy_load(blob_path+"/weight_fw.npy");
    cnpy::NpyArray arr_weight_bw = cnpy::npy_load(blob_path+"/weight_bw.npy");
    cnpy::NpyArray arr_output = cnpy::npy_load(blob_path+"/output.npy");
    cnpy::NpyArray arr_output_state_fw = cnpy::npy_load(blob_path+"/output_state_fw.npy");
    cnpy::NpyArray arr_output_state_bw = cnpy::npy_load(blob_path+"/output_state_bw.npy");

    dump_npy_array_info(arr_init_state_fw,  "init_st_fw: ");
    dump_npy_array_info(arr_init_state_bw,  "init_st_bw: ");
    dump_npy_array_info(arr_input,          "input     : ");
    dump_npy_array_info(arr_weight_fw,      "weight_fw : ");
    dump_npy_array_info(arr_weight_bw,      "weight_bw : ");
    dump_npy_array_info(arr_output,         "output    : ");
    dump_npy_array_info(arr_output_state_fw,"out_st_fw : ");
    dump_npy_array_info(arr_output_state_bw,"out_st_bw : ");

    float * output     = new float[STEP_LENGTH * 2*HIDDEN_SIZE];
    float * out_state_fw  = new float[NUM_LAYERS*2*HIDDEN_SIZE];
    float * out_state_bw  = new float[NUM_LAYERS*2*HIDDEN_SIZE];
    float * output_0    =  arr_output.data<float>();
    float * out_state_fw_0 =  arr_output_state_fw.data<float>();
    float * out_state_bw_0 =  arr_output_state_bw.data<float>();

    float * input = arr_input.data<float>();
    float * init_state_fw = arr_init_state_fw.data<float>();
    float * init_state_bw = arr_init_state_bw.data<float>();
    float * weight_fw = arr_weight_fw.data<float>();
    float * weight_bw = arr_weight_bw.data<float>();

    rnn_type rnn;
    init_dynamic_rnn(&rnn, NUM_LAYERS, STEP_LENGTH, INPUT_SIZE, HIDDEN_SIZE, true, FORGET_BIAS,
       init_state_fw, init_state_bw, weight_fw, weight_bw);

    stack_bidirectional_dynamic_rnn(&rnn, input, output, out_state_fw, out_state_bw);

    destroy_dynamic_rnn(&rnn);

    std::cout<<"------ origin out_stat fw (layers, 2*hidden_size)"<<std::endl;
    for(int j=0;j<NUM_LAYERS;j++){
        for(int i=0;i<2*HIDDEN_SIZE;i++){
            std::cout<<out_state_fw_0[j*2*HIDDEN_SIZE + i]<<", ";
        }
        std::cout<<std::endl;
    }
    std::cout<<"------ origin out_stat bw (layers, 2*hidden_size)"<<std::endl;
    for(int j=0;j<NUM_LAYERS;j++){
        for(int i=0;i<2*HIDDEN_SIZE;i++){
            std::cout<<out_state_bw_0[j*2*HIDDEN_SIZE + i]<<", ";
        }
        std::cout<<std::endl;
    }
    std::cout<<"------ origin out (seq, 2*hidden_size or hiddensize)"<<std::endl;
    for(int j=0;j<STEP_LENGTH;j++){
        for(int i=0;i<2*HIDDEN_SIZE;i++){
            std::cout<<output_0[j*2*HIDDEN_SIZE + i]<<", ";
        }
        std::cout<<std::endl;
    }
    std::cout<<std::endl;

    std::cout<<"++++++ cal out_stat fw"<<std::endl;
    for(int j=0;j<NUM_LAYERS;j++){
        for(int i=0;i<2*HIDDEN_SIZE;i++){
            std::cout<<out_state_fw[j*2*HIDDEN_SIZE + i]<<", ";
        }
        std::cout<<std::endl;
    }
    std::cout<<"++++++ cal out_stat bw"<<std::endl;
    for(int j=0;j<NUM_LAYERS;j++){
        for(int i=0;i<2*HIDDEN_SIZE;i++){
            std::cout<<out_state_bw[j*2*HIDDEN_SIZE + i]<<", ";
        }
        std::cout<<std::endl;
    }
    std::cout<<"++++++ cal out"<<std::endl;
    for(int j=0;j<STEP_LENGTH;j++){
        for(int i=0;i<2*HIDDEN_SIZE;i++){
            std::cout<<output[j*2*HIDDEN_SIZE + i]<<", ";
        }
        std::cout<<std::endl;
    }
    std::cout<<std::endl;

    delete [] output;
    delete [] out_state_fw;
    delete [] out_state_bw;

    return 0;
}