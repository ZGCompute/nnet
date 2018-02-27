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

#ifndef _OPTIMIZER
#define _OPTIMIZER
#include "optimizer.cpp"

class optimizer {

  
public:

  
  // init Constructor and destructor
  optimizer();
 ~optimizer();


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
  map< string, vector< cv::Mat > > optimizer::update_parameters_gd( map< string, vector< cv::Mat > > parameteters, \
						       map< string, vector< cv::Mat > > grads, float learning_rate );


  // Creates a list of random minibatches from (X, Y)  
  // Arguments:
  // X -- input data, of shape (input size, number of examples)
  // Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
  // mini_batch_size -- size of the mini-batches, integer
  // Returns:
  // mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
  map< string, vector< cv::Mat > > optimizer::rand_mini_batches( vector< cv::Mat > X, vector< cv::Mat > Y, \
						         	    int mini_batch_size = 64, int seed = 0 );
  

};

#endif
