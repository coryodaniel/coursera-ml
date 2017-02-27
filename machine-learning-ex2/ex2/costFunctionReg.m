function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

m = length(y); % number of training examples

theta0 = [0; theta(2:end)];

hyp = sigmoid(X * theta);
reg_term = lambda / (2*m) * theta0' * theta0;
J = -1 * ((1/m) * sum(y .* log(hyp) + (1 - y) .* log(1 - hyp))) + reg_term;

grad = (1/m) * (X' * (hyp - y) + lambda * theta0);

% =============================================================

end
