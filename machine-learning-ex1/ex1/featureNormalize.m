function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

m = size(X, 1);
mu = mean(X, 1);
sigma = std(X, 0, 1); % N-1, 1st dimension; sigma can be RANGE or STDDEV

%X_norm = (x_ij - mu_ij) / sigma_ij
for i = 1:rows(X)
  for j = 1:columns(X)
    X_norm(i,j) = (X(i,j) - mu(1,j)) / sigma(1,j);
  end
end

% This could be optimized w/ repmat maybe?

end
