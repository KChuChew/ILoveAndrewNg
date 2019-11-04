function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% calculate linear regression cost
unregularized_cost = (sum((X * theta - y).^2)) / (2 * m);
regularization_term = sum(theta(2:end).^2) * lambda / (2 * m);

J = unregularized_cost + regularization_term;

% calculate linear regression gradient
error_vector = X * theta - y;

% sum(h(X(i) - y(i)) * x(i)j)
unregularized_gradient = X' * error_vector / m;
grad = unregularized_gradient;

% + theta(j) * lambda / m
grad(2:end) = grad(2:end) + (theta(2:end) * lambda / m);

% =========================================================================

grad = grad(:);

end
