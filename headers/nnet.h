#include <stdio.h>
#include <iostream>
#include <fstream>
#include <strstream>

#include <vector>
#include <string>
#include <map>
#include <list>
#include <math.h>

#include <opencv/cv.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include "initializer.hpp"
#include "optimizer.hpp"

using namespace std;

#ifndef NNET_HPP
#define NNET_HPP

/*********************************************************************************************************/
/* Deep fully-connected neural network, Deep Convolutional network, and Deep Recurrent network           */ 
/* implementation using OpenCV & OpenCL backend. Designed to support deep learning with NIFTI data.      */
/* compile with: "g++ nnet.cpp -o nnet -I headers/ -std=c++11 `pkg-config opencv --cflags --libs`"       */
/*********************************************************************************************************/

class nnet {

public:
 
  // Constructor and destructor
  nnet();
  virtual ~nnet();
  
  // The initializer object
  initializer* init;
  
  // The optimizer object
  optimizer* opt;

 /*********************************************************************************************************/
 /* General purpose containers for Activations and cached parameters                                      */
 /* at any given layer in the model (Fully-connected, CNN, or RNN)                                        */
 /*********************************************************************************************************/

 // Container for fully connected activations and cache
 struct Activations{

    cv::Mat A;
    map< string, map< string, cv::Mat > > cache;

  };
  
  // store the activations/gradients cache at given layer in the model
  // and the final output AL
  struct ActivationsL{

    cv::Mat AL;
    vector< map< string, map< string, cv::Mat > > > caches;

  };
  
  // Struct for cnn layers:
  // store the activations/gradients cache at a given layer in the model
  // and the final output AL ( 3d volume )
  struct cacheActivationsL3D {

    vector< cv::Mat > AL;
    map< string, vector< cv::Mat > > caches;
  };

  // Struct for Rnn layers:
  // Store the activations/predictions
  // cached inputs ( a_prev, xt) and parameters
  // at time t in the sequnce
  struct cacheActivationsRNN {

    vector< cv::Mat > a_next;
    vector< cv::Mat > yt_pred;
    map< string, vector< cv::Mat > > cache;
    map< string, vector< cv::Mat > > parameters;

  };

  // Struct for Rnn models:
  // Store the activations/predictions
  // cached inputs ( a_prev, xt) and parameters
  // at time t in the sequnce
  struct cacheActivationsLRNN {

    vector< cv::Mat > a;
    vector< cv::Mat > y_pred;
    vector< map< string, map< string, cv::Mat > > > caches;
    vector< map< string, map< string, cv::Mat > > > parameters;

  };

  // Container for LSTM model outputs.
  // Contains: 
  // a_next -- next hidden state, of shape (n_a, m)
  // c_next -- next memory state, of shape (n_a, m)
  // yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
  // cache -- tuple of values needed for the backward pass, contains (a_next, c_next, a_prev, c_prev, xt, parameters)
  struct cacheActivationsLSTM {

    vector< cv::Mat > a_next;
    vector< cv::Mat > a_next;
    vector< cv::Mat > yt_pred;
    map< string, vector< cv::Mat > > cache;
    map< string, vector< cv::Mat > > parameters;

  };
  
private:

  /*********************************************************************************************************/
  /*                                                                                                       */
  /* General purpose I/O for image train/dev/test data                                                     */
  /*                                                                                                       */
  /*********************************************************************************************************/
  
  // Pointer to image data we read into memory from NIFTI
  vector< cv::Mat >* imgData;
  
  // Loads 3d volumetric data into memory from NIFTI (*.nii) binary file format
  // Agruments:
  // fname -- the string of the .nii file name to load into memory
  void load_img3d( string fname );

  /*********************************************************************************************************/
  /*                                                                                                       */
  /* Fully connected Deep-NN primatives                                                                    */
  /*                                                                                                       */
  /*********************************************************************************************************/
  
  // Implements the forward propagation for the LINEAR->ACTIVATION layer
  // Arguments:
  // A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
  // W -- weights matrix: array of shape (size of current layer, size of previous layer)
  // b -- bias vector, array of shape (size of the current layer, 1)
  // l -- current layer index for storing caches
  // activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
  // Returns:
  // A -- the output of the activation function, also called the post-activation value 
  // cache -- a dictionary containing "linear_cache" and "activation_cache";
  // stored for computing the backward pass efficiently
  Activations linear_activation_forward( cv::Mat A_prev, cv::Mat W, cv::Mat b, string activation, int l );
  
  // compute the linear portion of the activation at layer l
  // by the dot product of W * A, add bias
  cv::Mat linear_forward( cv::Mat A, cv::Mat W, cv::Mat b );
  
  // kernel method for computing the sigmoid activation
  // at a given node with basis function formed from
  // input data, weights and bias nodes
  cv::Mat sigmoid( cv::Mat Z );
  
  // kernel method for computing the relu activation
  // at a given node with basis function formed from
  // input data, weights and bias nodes
  cv::Mat relu_fc( cv::Mat Z );

  // Kernel method for computing the softmax activation
  // at a collection of nodes in a given layer
  vector< cv::Mat > softmax( vector< cv::Mat > Z );
  
  // Forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
  // Arguments:
  // X -- data, array of shape (input size, number of examples)
  // parameters -- output of initialize_parameters_deep()
  // Returns:
  // AL -- last post-activation value
  // caches -- list of caches containing:
  // every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
  // the cache of linear_sigmoid_forward() (there is one, indexed L-1)
  ActivationsL L_model_forward( cv::Mat X, map< string, cv::Mat > parameters );

  // The cross-entropy cost function for sigmoid output
  // Arguments:
  // AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
  // Y -- true "label" vector (for example: containing 0 if non-id, 1 if id), shape (1, number of examples)
  // Returns:
  // cost -- cross-entropy cost
  double compute_cost( cv::Mat AL, cv::Mat Y );
  
  // Fully connected nnet prototypes:
  // The linear portion of backward propagation for a single layer (layer l)
  // Arguments:
  // dZ -- Gradient of the cost with respect to the linear output (of current layer l)
  // cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
  // Returns:
  // dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
  // dW -- Gradient of the cost with respect to W (current layer l), same shape as W
  // db -- Gradient of the cost with respect to b (current layer l), same shape as b
  map< string, cv::Mat > linear_backward( cv::Mat dZ, map< string, cv::Mat> cache );
                                                                                                        
  /*********************************************************************************************************/
  /*                                                                                                       */
  /* CNN primatives for 2D and 3D models:                                                                  */
  /*                                                                                                       */
  /*********************************************************************************************************/
  
  // Pad an input image with zeros in x and y dimensions
  // Arguments:
  // X -- an input 2D cv::Mat image
  // pad_size -- the number of zeros to pad at the beginning and end of every row, column
  // Returns:
  // padded_image -- a cv::Mat representing the input image padded by pad_size
  vector< cv::Mat > zero_pad( vector< cv::Mat > X, int pad_size );

  // Unroll an N-dimensional image stored in a vector of cv::Mats
  // Arguments:
  // img -- a vector of Nd cv::Mats
  // Returns:
  // unrolled img -- a vector containing 1 1d cv::Mat with the unrolled data
  vector< cv::Mat > img_unroll( vector< cv::Mat > img );
 
  // FIXME: check to make sure elemts of s are summed along the correct axis
  // Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation 
  // of the previous layer.
  // Arguments:
  // a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
  // W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
  // b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
  // Returns:
  // Z -- a scalar value, result of convolving the sliding window (W, b) on a slice x of the input data
  // Element wise product of a_slice * W. then add bias
  cv::Scalar conv_single_step( cv::Mat a_slice_prev, cv::Mat W, cv::Mat b );
 
  // Implements the forward propagation for a convolution function    
  // Arguments:
  // A_prev -- output activations of the previous layer, array of shape (m, n_H_prev, n_W_prev, n_C_prev)
  // W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
  // b -- Biases, numpy array of shape (1, 1, 1, n_C)
  // hparameters -- dictionary containing "stride" and "pad"
  // Returns:
  // Z -- conv output, array of shape (m, n_H, n_W, n_C)
  // cache -- cache of values needed for the conv_backward() function
  cacheActivationsL3D conv_forward( vector<cv::Mat> A_prev, vector<cv::Mat> W, vector<cv::Mat> b, map< string, int > hparameters );

  // Implements the forward pass of the pooling layer
  // Arguments:
  // A_prev -- Input data, array of shape (m, n_H_prev, n_W_prev, n_C_prev)
  // hparameters -- dictionary containing "f" and "stride"
  // mode -- the pooling mode you would like to use, defined as a string ("max" or "average")  
  // Returns:
  // A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
  // cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters 
  cacheActivationsL3D pool_forward( vector< cv::Mat > A_prev, map< string, int > hparameters, string mode );

  // Implements the backward pass of the pooling layer
  // Arguments:
  // dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
  // cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters 
  // mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
  // Returns:
  // dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
  vector< cv::Mat >  pool_backward( vector<cv::Mat> dA, nnet::cacheActivationsL3D cache_actvL, string mode );

  // Implement the backward propagation for a convolution function
  // Arguments:
  // dZ -- gradient of the cost with respect to the output of the conv layer (Z), array of shape (m, n_H, n_W, n_C)
  // cache -- cache of values needed for the conv_backward(), output of conv_forward()
  // Returns:
  // dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev), array of shape (m, n_H_prev, n_W_prev, n_C_prev)
  // dW -- gradient of the cost with respect to the weights of the conv layer (W) array of shape (f, f, n_C_prev, n_C)
  // db -- gradient of the cost with respect to the biases of the conv layer (b) array of shape (1, 1, 1, n_C)
  map< string, vector< cv::Mat > > conv_backward( vector<cv::Mat> dZ, cacheActivationsL3D cache_actvL );

  // Creates a mask from an input matrix x, to identify the max entry of x.
  // Arguments:
  // x -- Array of shape (f, f)
  // Returns:
  // mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
  cv::Mat create_mask_from_window( cv::Mat x );
 
  // Distributes the input value in the matrix of dimension shape
  // Arguments:
  // dz -- input scalar
  // shape -- the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz
  // Returns:
  // a -- Array of size (n_H, n_W) for which we distributed the value of dz
  cv::Mat distribute_value( cv::Scalar dZ, int n_H, int n_W );
                                                                                                         
  /*********************************************************************************************************/
  /*                                                                                                       */
  /* RNN primatives for sequence models                                                                    */
  /*                                                                                                       */
  /*********************************************************************************************************/

  // Implements a single forward step of the vanilla RNN-cell 
  // Arguments:
  // xt -- your input data at timestep "t", array of shape (n_x, m).
  // a_prev -- Hidden state at timestep "t-1", array of shape (n_a, m)
  // parameters -- dictionary containing:
  // Wax -- Weight matrix multiplying the input, array of shape (n_a, n_x)
  // Waa -- Weight matrix multiplying the hidden state, array of shape (n_a, n_a)
  // Wya -- Weight matrix relating the hidden-state to the output, array of shape (n_y, n_a)
  // ba --  Bias, array of shape (n_a, 1)
  // by -- Bias relating the hidden-state to the output, array of shape (n_y, 1)
  // Returns:
  // a_next -- next hidden state, of shape (n_a, m)
  // yt_pred -- prediction at timestep "t", array of shape (n_y, m)
  // cache -- tuple of values needed for the backward pass, contains (a_next, a_prev, xt, parameters)
  cacheActivationsRNN rnn_cell_forward( vector< cv::Mat>  xt, vector< cv::Mat > a_prev, map< string, vector< cv::Mat > > );  
  
  // Implement the forward propagation of the recurrent neural network 
  // Arguments:
  // x -- Input data for every time-step, of shape (n_x, m, T_x).
  // a0 -- Initial hidden state, of shape (n_a, m)
  // parameters -- python dictionary containing:
  // Waa -- Weight matrix multiplying the hidden state, array of shape (n_a, n_a)
  // Wax -- Weight matrix multiplying the input, array of shape (n_a, n_x)
  // Wya -- Weight matrix relating the hidden-state to the output, array of shape (n_y, n_a)
  // ba --  Bias array of shape (n_a, 1)
  // by -- Bias relating the hidden-state to the output, array of shape (n_y, 1)
  // Returns:
  // a -- Hidden states for every time-step, array of shape (n_a, m, T_x)
  // y_pred -- Predictions for every time-step, array of shape (n_y, m, T_x)
  // caches -- tuple of values needed for the backward pass, contains (list of caches, x)
  cacheActivationsLRNN rnn_forward( vector< cv::Mat > x, vector< cv::Mat > a0, map< string, vector< cv::Mat > > parameters );

  // Implement a single forward step of the LSTM-cell 
  // Arguments:
  // xt -- your input data at timestep "t", array of shape (n_x, m).
  // a_prev -- Hidden state at timestep "t-1", array of shape (n_a, m)
  // c_prev -- Memory state at timestep "t-1", array of shape (n_a, m)
  // parameters -- python dictionary containing:
  // Wf -- Weight matrix of the forget gate, array of shape (n_a, n_a + n_x)
  // bf -- Bias of the forget gate, array of shape (n_a, 1)
  // Wi -- Weight matrix of the update gate, array of shape (n_a, n_a + n_x)
  // bi -- Bias of the update gate, array of shape (n_a, 1)
  // Wc -- Weight matrix of the first "tanh", array of shape (n_a, n_a + n_x)
  // bc --  Bias of the first "tanh", array of shape (n_a, 1)
  // Wo -- Weight matrix of the output gate, array of shape (n_a, n_a + n_x)
  // bo --  Bias of the output gate, array of shape (n_a, 1)
  // Wy -- Weight matrix relating the hidden-state to the output, array of shape (n_y, n_a)
  // by -- Bias relating the hidden-state to the output,  array of shape (n_y, 1)                   
  // Returns:
  // a_next -- next hidden state, of shape (n_a, m)
  // c_next -- next memory state, of shape (n_a, m)
  // yt_pred -- prediction at timestep "t", array of shape (n_y, m)
  // cache -- tuple of values needed for the backward pass, contains (a_next, c_next, a_prev, c_prev, xt, parameters)
  cacheActivationsLSTM lstm_cell_forward( vector< cv::Mat> xt, vector< cv::Mat > a_prev, vector< cv::Mat > c_prev, map< string, vector< cv::Mat > parameters );
  
};

#endif
