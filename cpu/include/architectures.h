#ifndef CNN_ARCHITECTURES_H
#define CNN_ARCHITECTURES_H


// C++
#include <list>
#include <fstream>
// self
#include "pipeline.h"


namespace architectures {
    using namespace pipeline;

    // Random initialization, C++, the generated number is too large, hundreds before softmax, it is directly exploded, it is a rip off
    extern data_type random_times;

    // Global variables, whether to backward, and access speed should be slower
    extern bool no_grad;

    // Turn off gradient-dependent calculations within scope
    class WithoutGrad final {
    public:
        explicit WithoutGrad() {
            architectures::no_grad = true;
        }
        ~WithoutGrad() noexcept {
            architectures::no_grad = false;
        }
    };

    // Used to unify various data types, but doing so will introduce virtual function polymorphism, and the efficiency will not have a big impact on forward backward, save_weights
    // backward cannot be const, because relu operates in-place.
    class Layer {
    public:
        const std::string name;  // The name of this layer
        std::vector<tensor> output;  // Output tensor
    public:
        Layer(std::string& _name) : name(std::move(_name)) {}
        virtual std::vector<tensor> forward(const std::vector<tensor>& input) = 0;
        virtual std::vector<tensor> backward(std::vector<tensor>& delta) = 0;
        virtual void update_gradients(const data_type learning_rate=1e-4) {}
        virtual void save_weights(std::ofstream& writer) const {}
        virtual void load_weights(std::ifstream& reader) {}
        virtual std::vector<tensor> get_output() const { return this->output; }
    };


    class Conv2D: public Layer {
    private:
        // Intrinsic information of the convolutional layer
        std::vector<tensor> weights; // weight parameters of the convolution kernel, out_channels X in_channels X kernel_size X kernel_size
        std::vector<data_type> bias; // bias (can be written as tensor1D)
        const int in_channels; // The feature map to be filtered has several channels
        const int out_channels; // This layer of convolution has several convolution kernels
        const int kernel_size; // side length of the convolution kernel
        const int stride; // Convolution step size
        const int params_for_one_kernel; // The number of parameters of a convolution kernel
        const int padding = 0; // padding padding amount, this will break this fragile program and sacrifice performance. We'll talk about this later when we have time
        std::default_random_engine seed; // Initialized seed
        std::vector<int> offset; // Offset of convolution, auxiliary
        // Historical information
        std::vector<tensor> __input; // Need to find the gradient, in fact it stores a pointer
        // Buffer, avoid each reallocation
        std::vector<tensor> delta_output; // Store the gradient passed back to the previous layer
        std::vector<tensor> weights_gradients; // Gradient of weights
        std::vector<data_type> bias_gradients; // gradient of bias
    public:
        Conv2D(std::string _name, const int _in_channels=3, const int _out_channels=16, const int _kernel_size=3, const int _stride=2);
        // Forward process of convolution operation, batch_num X in_channels X H X W
        std::vector<tensor> forward(const std::vector<tensor>& input);
        // For optimization, put some data on the heap into the stack area, local variables are fast
        std::vector<tensor> backward(std::vector<tensor>& delta);
        // Update gradient
        void update_gradients(const data_type learning_rate=1e-4);
        // Save the value
        void save_weights(std::ofstream& writer) const;
        // Load authority value
        void load_weights(std::ifstream& reader);
        // Get the parameter values of this convolutional layer
        int get_params_num() const;
        };
    

    class MaxPool2D: public Layer {
    private:
        // Intrinsic properties of this layer
        const int kernel_size;
        const int step;
        const int padding; // Not supported yet
        // Buffer, avoid each reallocation
        std::vector< std::vector<int> > mask; // Record which locations have gradients passed back, the bth graph, one std::vector<int> per graph
        std::vector<tensor> delta_output; // returned delta
        std::vector<int> offset; // offset pointer, the same as the previous Conv2D
    public:
        MaxPool2D(std::string _name, const int _kernel_size=2, const int _step=2)
        : Layer(_name), kernel_size(_kernel_size), step(_step), padding(0),
        offset(_kernel_size * _kernel_size, 0) {}
        // Forward
        std::vector<tensor> forward(const std::vector<tensor>& input);
        // Backpropagation
        std::vector<tensor> backward(std::vector<tensor>& delta);
    
    };


    class ReLU : public Layer  {
    public:
        ReLU(std::string _name) : Layer(_name) {}
        std::vector<tensor> forward(const std::vector<tensor>& input);
        std::vector<tensor> backward(std::vector<tensor>& delta);
    };


    // Linear transformation layer
    class LinearLayer: public Layer {
    private:
        // Intrinsic information of the linear layer
        const int in_channels; // Number of neurons input
        const int out_channels; // Number of neurons output
        std::vector<data_type> weights; // Weight matrix (this can actually be changed to Tensor1D, the data types can be unified, but the weights_gradients at the end are not easy to use)
        std::vector<data_type> bias; // bias
        // Historical information
        std::tuple<int, int, int> delta_shape; // Write down the shape of delta, from 1 X 4096 to 128 * 4 * 4
        std::vector<tensor> __input; // When passing the gradient back, you need to input Wx + b, and you need to keep x
        // The following is the buffer
        std::vector<tensor> delta_output; // delta returns the gradient to the input
        std::vector<data_type> weights_gradients; // cache area, gradient of the weight matrix
        std::vector<data_type> bias_gradients; // gradient of bias
    public:
        LinearLayer(std::string _name, const int _in_channels, const int _out_channels);
        // Do Wx + b matrix operations
        std::vector<tensor> forward(const std::vector<tensor>& input);
        std::vector<tensor> backward(std::vector<tensor>& delta);
        void update_gradients(const data_type learning_rate=1e-4);
        void save_weights(std::ofstream& writer) const;
        void load_weights(std::ifstream& reader);
        };


    // This BatchNorm is made in a different channel. I don't know if it's implemented in detail, but I have time to fill in the gaps later
    // Currently only consider BN of Conv layer
    class BatchNorm2D: public Layer {
    private:
    // Inherent information
        const int out_channels;
        const data_type eps;
        const data_type momentum;
        // Parameters to learn (use vector directly here, it's not a problem if it's not uniform)
        std::vector<data_type> gamma;
        std::vector<data_type> beta;
        // Historical information to keep
        std::vector<data_type> moving_mean;
        std::vector<data_type> moving_var;
        // Buffer, avoid each reallocation
        std::vector<tensor> normalized_input;
        std::vector<data_type> buffer_mean;
        std::vector<data_type> buffer_var;
        // Reserved gradient information
        std::vector<data_type> gamma_gradients;
        std::vector<data_type> beta_gradients;
        // Temporary gradient information, which is actually also a buffer
        tensor norm_gradients;
        // Need to use to find the gradient
        std::vector<tensor> __input;
    public:
        BatchNorm2D(std::string _name, const int _out_channels, const data_type _eps=1e-5, const data_type _momentum=0.1);
        std::vector<tensor> forward(const std::vector<tensor>& input);
        std::vector<tensor> backward(std::vector<tensor>& delta);
        void update_gradients(const data_type learning_rate=1e-4);
        void save_weights(std::ofstream& writer) const;
        void load_weights(std::ifstream& reader);
    };


    // Currently only the Conv layer is supported, but dropout is usually placed in the linear linear connection layer
     // Although the training can be trained normally, the test is a bit rubbish
    class Dropout: public Layer {
    private:
    // Inherent properties
        data_type p;
        int selected_num;
        std::vector<int> sequence;
        std::default_random_engine drop;
    // backward needs to be used
        std::vector<int> mask;
    public:
        Dropout(std::string _name, const data_type _p=0.5): Layer(_name), p(_p), drop(1314) {}
        std::vector<tensor> forward(const std::vector<tensor>& input);
        std::vector<tensor> backward(std::vector<tensor>& delta);
    };



    // A random CNN network structure that can run, not the real AlexNet
    class AlexNet {
    public:
        bool print_info = false;
    private:
        std::list< std::shared_ptr<Layer> > layers_sequence;
    public:
        AlexNet(const int num_classes=3, const bool batch_norm=false);
        // Forward
        std::vector<tensor> forward(const std::vector<tensor>& input);
        // Gradient retransmission
        void backward(std::vector<tensor>& delta_start);
        // Gradient updated to weight
        void update_gradients(const data_type learning_rate=1e-4);
        // Save the model
        void save_weights(const std::filesystem::path& save_path) const;
        // Load the model
        void load_weights(const std::filesystem::path& checkpoint_path);
        // GradCam visualization
        cv::Mat grad_cam(const std::string& layer_name) const;
    };
}



#endif //CNN_ARCHITECTURES_H
