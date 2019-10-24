function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% NO FOR LOOPS MOTHERFUCKER
% START UNREGULARIZED COST FUNCTION CALCULATION
num_labels_identity = eye(num_labels);

% Set up output matrix for multiclass 
training_labels_matrix = num_labels_identity(y', :);

% Add ones to the X data matrix
X = [ones(m, 1) X];
hidden_layer_activations = sigmoid(X * Theta1');
% Add ones to the hidden layer activations
hidden_layer_activations = [ones(m, 1) hidden_layer_activations];
output_layer_activations = sigmoid(hidden_layer_activations * Theta2');

log_output_activations = log(output_layer_activations);

negative_labels_matrix = 1 - training_labels_matrix;

negative_log_output_activations = log(1 - output_layer_activations);

positive_op = training_labels_matrix .* log_output_activations;
positive_op = sum(positive_op, 2);
negative_op = negative_labels_matrix .* negative_log_output_activations;
negative_op = sum(negative_op, 2);

final_op = positive_op + negative_op;
J = sum(final_op) / m * -1;
% END UNREGULARIZED COST FUNCTION CALCULATION

% START REGULARIZED COST FUNCTION
regularize_Theta1 = Theta1(:, (2:end));
regularize_Theta2 = Theta2(:, (2:end));

sq_Theta1_terms = regularize_Theta1.^2;
sq_Theta2_terms = regularize_Theta2.^2;

sum_Theta1 = sum(sq_Theta1_terms);
sum_Theta1 = sum(sum_Theta1);

sum_Theta2 = sum(sq_Theta2_terms);
sum_Theta2 = sum(sum_Theta2);

sums = sum_Theta1 + sum_Theta2;

J = J + ((sums * lambda) / (2 * m));
% END REGULARIZED COST FUNCTION


% NOTES FOR VECTORIZED IMPLEMENTATION OF NN TRAINING

% d(3) = a(3) - y
output_layer_deltas = output_layer_activations - training_labels_matrix;

% Steps for d(2) = Theta(2)' * d(3) .* g(prime)(z(2))
% z(2)
hidden_layer_inputs = X * Theta1';
% g(prime)(z(2))
sigmoid_gradients = sigmoid(hidden_layer_inputs) .* (1 - sigmoid(hidden_layer_inputs));
% d(2) = Theta(2)' * d(3) * sigmoid_gradients
hidden_layer_deltas = output_layer_deltas * Theta2(:, 2: end) .* sigmoid_gradients;

Theta1_grad = hidden_layer_deltas' * X;
Theta2_grad = output_layer_deltas' * hidden_layer_activations;

Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
