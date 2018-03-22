// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Copyright 2018 Zachary Greenberg

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// File:    initializer.hpp
// Author:  Zachary Greenberg
// Summary: Class initialization layer providing support 
//          for initializing weight vectors with zeros, random values,
//          and He. Initialization is included.

// Inspired by https://www.coursera.org/learn/convolutional-neural-networks 
// & https://arxiv.org/pdf/1502.01852.pdf

#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <list>
#include <math.h>

#include <opencv/cv.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

using namespace std;

#ifndef INITIALIZER_HPP
#define INITIALIZER_HPP

// Deep fully-connected neural network, and Deep Convolutional netowrk implementation using OpenCV & OpenCL backend 
// compile with: "g++ nnet.cpp -o nnet -I headers/ -std=c++11 `pkg-config opencv --cflags --libs`"

class initializer {
  
public:
  
  // init Constructor and destructor
  initializer();
  virtual ~initializer();

private:

  // Arguments:
  // layer_dims -- 2d vector containing the size of each layer.
  // Returns:
  // parameters -- dictionary (map) containing your parameters
  //               as c++ vectors of 3d (for conv layers) or 2d (for fc layers)
  //               cv::Mat "W1", "b1", ..., "WL", "bL":
  // Fully connected example wieghts:
  // W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
  // b1 -- bias vector of shape (layers_dims[1], 1)
  // WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
  // bL -- bias vector of shape (layers_dims[L], 1)
  // FIXME: add example of conv2d wight shapes
  map< string, vector< cv::Mat > > parameters 
  init_params_he( vector< vector< int > > layer_dims, vector< string > layer_types ); 

  // Arguments:
  // layer_dims -- 2d vector containing the size of each layer.
  // Returns:
  // parameters -- dictionary (map) containing your parameters
  //               as c++ vectors of 3d (for conv layers) or 2d (for fc layers)
  //               cv::Mat "W1", "b1", ..., "WL", "bL":
  // Fully connected example wieghts:
  // W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
  // b1 -- bias vector of shape (layers_dims[1], 1)
  // WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
  // bL -- bias vector of shape (layers_dims[L], 1)
  // FIXME: add example of conv2d wight shapes
  map< string, vector< cv::Mat > > parameters 
  init_params_rand( vector< vector< int > > layer_dims, vector< string > layer_types );

  // Arguments:
  // layer_dims -- 2d vector containing the size of each layer.
  // Returns:
  // parameters -- dictionary (map) containing your parameters
  //               as c++ vectors of 3d (for conv layers) or 2d (for fc layers)
  //               cv::Mat "W1", "b1", ..., "WL", "bL":
  // Fully connected example wieghts:
  // W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
  // b1 -- bias vector of shape (layers_dims[1], 1)
  // WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
  // bL -- bias vector of shape (layers_dims[L], 1)
  // FIXME: add example of conv2d wight shapes
  map< string, vector< cv::Mat > > parameters 
  init_params_zeros( vector< vector< int > > layer_dims, vector< string > layer_types );
  
  // Implements a n-layer neural network: LINEAR->RELU->LINEAR->RELU->....LINEAR->SIGMOID.
  // or: CONV->MAXPOOL->RELU->CONV->MAXPOOL->RELU->...LINEAR->SOFTMAX 
  // Arguments:
  // X -- input data, of shape (number of examples,?)
  // Y -- true "label" vector (containing 0 for EXAMPLE CLASS; 1 for EAXMPLE2 CLASS), of shape (1, number of examples)
  // learning_rate -- learning rate for gradient descent 
  // num_iterations -- number of iterations to run gradient descent
  // print_cost -- if True, print the cost every 1000 iterations
  // initialization -- flag to choose which initialization to use ("zeros","random" or "he")
  // Returns:
  // parameters -- parameters learnt by the model
  map< string, vector< cv::Mat > > parameters \
  model( vector< cv::Mat > X, vector< cv::Mat > Y, float learning_rate = 0.01, int num_iters= 15000, bool print_cost = True, string init = "he" );
  
};

#endif
