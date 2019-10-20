function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularize
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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


% calculate cost without regularize term
theta_matrix = repmat(theta', size(X, 1), 1); % copies theta row-wise X.row times (# training samples)

hypothesis_sums = sum(theta_matrix .* X, 2); % sum(Matrix, 2) sums along rows

sigmoid_hypothesis = sigmoid(hypothesis_sums);

positive_log_hypothesis = log(sigmoid_hypothesis);

positive_hypothesis = -y' * positive_log_hypothesis;

negative_log_hypothesis = log(1 - sigmoid_hypothesis);

negative_hypothesis = (1 - y)' * negative_log_hypothesis;

non_regularize_cost = (positive_hypothesis - negative_hypothesis) / m;

% regularize term
regularize_theta = theta(2 : end); % theta 1 to end (don't regularize theta(0))
%disp(size(theta));
%disp(size(regularize_theta));

squared_regularize_theta = regularize_theta.^2;

%disp(regularize_theta(1));
%disp(squared_regularize_theta(1));
sum_sqrd_reg_theta = sum(squared_regularize_theta);
%disp(sum_sqrd_reg_theta);
regularize_term = (lambda * sum_sqrd_reg_theta) / (2 * m);

J = non_regularize_cost + regularize_term; 

% calculate gradient 0
sigmoid_diff = sigmoid_hypothesis - y;
x0_terms = X(:, 1);
%disp(size(sigmoid_y_diff));
%disp(size(x0_terms));
x0_mult_sigmoid_diff = sigmoid_diff .* x0_terms;
grad(1) = sum(x0_mult_sigmoid_diff) / m;

% calculate gradient 1 to n
x_without_x0 = X(:, 2:end);
%disp(size(x_without_x0));
non_regularize_grad_term = (sigmoid_diff' * x_without_x0) / m;
%disp(size(non_regularized_grad_term));
regularize_term = theta(2: end) * lambda / m;

grad(2: end) = non_regularize_grad_term' + regularize_term;

%disp(size(regularize_term));
%disp(size(grad(2: end)));

%grad(2: end) = grad(2: end);

%grad(1) = (sum(X(:, 1) .* (sigmoid_hypothesis - y), 2)) / m;
%disp(size(X(:, 1)));
%disp(size(sigmoid_hypothesis));
%disp(size(y));

% =============================================================

end
