clc; clear all; close all;
f = functions_();

% can see results with blur_sigma=6, iterations=8, lucy_iterations=7, sigma=1e-4
blur_sigma = 6;
[x, X, h, H, y, Y, n] = f.input_and_observations(blur_sigma);
blur_sigma_hat = 1.5*blur_sigma;
[~, ~, h0, ~, ~, ~, ~] = f.input_and_observations(blur_sigma_hat);
h_hat = h0 * 1;

lucy_iterations = 7;
iterations = 20;
for k=1:iterations,
    x_hat = deconvlucy(y, h_hat, lucy_iterations);
    %x_hat = deconvwnr(y, h_hat);
    %x_hat = deconvreg(y, h_hat);
    h_hat = deconvlucy(y, x_hat, lucy_iterations);
end

% Show results
figure; hold on
subplot(321)
f.show_image(x)
title('Ground truth x')

subplot(322)
f.show_image(h)
title('Kernel ground truth')

subplot(323)
f.show_image(y)
title('Observations y')

subplot(324)
f.show_image(h0)
title('Initial estimate for kernel')

subplot(325)
f.show_image(x_hat)
title('Estimation x\_hat')

subplot(326)
f.show_image(h_hat)
title('Kernel estimate')

linkaxes
