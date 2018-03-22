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

// File:    optimizer.hpp
// Author:  Zachary Greenberg
// Summary: Class optimization layer providing support for minimizing model
//          optimization objectives (least squares, cross-entropy, softmax) 
//          using several popular algorithms (Gradient-Descent, Stochastic Descent, 
//          Mini-batch Gradient Descent, RMS Prop, Adam Optimization)
//          Support for initializing weight vectors with zeros, random values,
//          and He. Initialization is included.

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

#ifndef OPTIMIZER_HPP_
#define OPTIMIZER_HPP_

class optimizer {

public:

  // Constructor and destructor
  optimizer();
  virtual ~optimizer();

private:

  // Update parameters using one step of gradient descent
  // Arguments:
  // parameters -- dictionary containing your parameters to be updated:
  // parameters['W' + str(l)] = Wl
  // parameters['b' + str(l)] = bl
  // grads -- dictionary containing your gradients to update each parameters:
  // grads['dW' + str(l)] = dWl
  // grads['db' + str(l)] = dbl
  // learning_rate -- the learning rate, scalar.
  // Returns:
  // parameters -- dictionary containing your updated parameters 
  map< string, vector< cv::Mat > > update_parameters_gd( map< string, vector< cv::Mat > > parameteters, \
						       map< string, vector< cv::Mat > > grads, float learning_rate );

  // Creates a list of random minibatches from (X, Y)  
  // Arguments:
  // X -- input data, of shape (input size, number of examples)
  // Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
  // mini_batch_size -- size of the mini-batches, integer
  // Returns:
  // mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
  map< string, vector< cv::Mat > > rand_mini_batches( vector< cv::Mat > X, vector< cv::Mat > Y, \
						         	    int mini_batch_size = 64, int seed = 0 );
  
};

#endif
