clc; clear all; close all;
f = functions_();

epsilon = 1e-2; % What's zero?
noise_sigma = 5e-2; % std. dev. of noise to add
max_its = 1e3; % iterations to perform
lambda = 0.1; % ell_2 regularization for x_grad step

blur_sigma = 2;
[x, X, h, H, y, Y, N] = f.input_and_observations(blur_sigma);

avg_n = 3;
h = ones(avg_n) ./ avg_n.^2;

% Vectorize observations and true image, make convolution of A a matrix (TODO:
% make A a function?
x = x(:);
y = y(:);
A = convmtx2(h, [N, N]);
mu = 1 / norm(A, 'fro').^2; % Step size.
y = A * x;

% Noise A -- we don't know it exactly
i = abs(A(:)) > epsilon;
A(i) = A(i) + noise_sigma*abs(randn(size(A(i))));

xk = x .* 0;
for k=1:max_its,
    xk1 = xk - mu*f.x_gradient(y, A, xk, lambda);
    xk = xk1;
end
