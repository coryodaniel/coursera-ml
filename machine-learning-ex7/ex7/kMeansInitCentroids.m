function centroids = kMeansInitCentroids(X, K)
%KMEANSINITCENTROIDS This function initializes K centroids that are to be 
%used in K-Means on the dataset X
%   centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
%   used with the K-Means on the dataset X
%

% You should return this values correctly
centroids = zeros(K, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should set centroids to randomly chosen examples from
%               the dataset X
%


% Randomly reorder the indices of examples
dataset_size = size(X,1);

% Generate randomized sorted vector of indicies (1-dataset_size)
all_randomized_ids = randperm(dataset_size);

select_random_ids = all_randomized_ids(1:K);

% X(ROWS, COLS)
centroids = X(select_random_ids,1);
% =============================================================

end

