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

% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

a1 = [ones(m,1), X];

z2 = a1 * Theta1';
a2 = [ones(size(z2,1),1), sigmoid(z2)];

z3 = a2 * Theta2';
a3 = sigmoid(z3);
hyp = a3;

% This will take the vector of answers 'y'
% and convert it into a matrix of 0s and 1s, where
% when the value was 10 in a element of the vector, then
% its value in the matrix will be [0 0 0 0 0 0 0 0 0 1]
% This makes it easy to get if the prediction was correct
% given ith row and looking for k classification...
% ie is this 10? [0 0 0 0 0 0 0 0 0 1](1, 10)... yes :D
yMatrix = eye(num_labels)(y,:);

% Compute non-regularized Cost "J"
% this works, though it's not vectorized...
for i = 1:m
  for k = 1:num_labels
    J += (
      yMatrix(i,k) * log(hyp(i,k)) + (1-yMatrix(i,k)) * log(1-hyp(i,k))
    );
  end
end
J = -(1/m) * J;

% regularization
theta1_squared = Theta1 .^ 2;
sum_theta1_squared = sum(theta1_squared(:,2:end)(:)); % 2:end to remove bias

theta2_squared = Theta2 .^ 2;
sum_theta2_squared = sum(theta2_squared(:,2:end)(:)); % 2:end to remove bias

J += (sum_theta1_squared + sum_theta2_squared) * lambda/(2*m);

% matrix of errors from OUTPUT node on each example
d3 = a3 - yMatrix;

% matrix of erros on HIDDEN node on each example
d2 = d3*Theta2(:,2:end).*sigmoidGradient(z2);

% Step 4: accumulate gradient into Delta^(l)
% In English:
%   * D2 is associated with each weight
%   * It is calculated by summing over each example, how far off was
%     that output neuron, multiplied by how lit up was the hidden neuron
%   * So if it was too low (on average), it will be negative
%   * This will dictate how we alter Theta2 for the next go-round
%   * Nicely, this summation over each example is accomplished via the
%     normal matrix dot-product
D2 = d3'*a2;
D1 = d2'*a1;

% Turn the sum of the errors over the examples (as discussed above)
% into the average error over the examples
% This is our new gradient
Theta2_grad += D2/m;
Theta1_grad += D1/m;

% Regularize that bit
Theta2_grad += lambda*Theta2/m;
Theta1_grad += lambda*Theta1/m;

% "Note that you should NOT be regularizing the first column of Θ(l)
% which is used for the bias term."
Theta2_grad(:,1) -= lambda*Theta2(:,1)/m;
Theta1_grad(:,1) -= lambda*Theta1(:,1)/m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
