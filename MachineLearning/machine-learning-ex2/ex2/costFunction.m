function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

%disp(size(theta));
%disp(size(X));
%disp(size(y));


% disp(sigmoid(0.001));
%disp(sigmoid(theta' * X(1, :)'));
% calculate J
theta_matrix = repmat(theta', size(X, 1), 1); % copies theta row-wise X.row times (# training samples)
%disp(theta_matrix(1, :));
%disp(theta_matrix(2, :));
hypothesis_sums = sum(theta_matrix .* X, 2); % sum(Matrix, 2) sums along rows
%disp(size(hypothesis_sums));
%disp(hypothesis_sums(1, :));
%disp(hypothesis_sums(2, :));
sigmoid_hypothesis = sigmoid(hypothesis_sums);
%disp(sigmoid_hypothesis);
positive_log_hypothesis = log(sigmoid_hypothesis);
%disp(size(log_hypothesis));
%disp(size(y));
positive_hypothesis = -y' * positive_log_hypothesis;

negative_log_hypothesis = log(1 - sigmoid_hypothesis);

negative_hypothesis = (1 - y)' * negative_log_hypothesis;

J = (positive_hypothesis - negative_hypothesis) / m;


% calculate gradient
grad = (X' * (sigmoid_hypothesis - y)) / m;

% =============================================================

end
