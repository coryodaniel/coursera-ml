function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta.
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    % multivariate convergence
    % thetaJ = thetaJ - alpha 1/m SUM( hypTheta(x_i) - y_i) * x_ji

    % instead of looping through, this can be done in place w/ transpose
    % hyp = X * theta % MxN * Nx1 % note, not x_i - yi) * x_ji
    % diff = hyp - y % (Mx1 - Mx1)
    %% for each example / feature x_ji
    % SUM( X' * diff ) % reverse the order and transpose, so we can do it in matrix math

    theta = theta - (alpha / m) * X' * (X * theta - y);

    % ============================================================

    % Save the cost J in every iteration
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
