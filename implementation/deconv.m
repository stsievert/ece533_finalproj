clc; clear all; close all;
f = functions_();

blur_sigma = 8;
[x, X, h, H, y, Y, n] = f.input_and_observations(blur_sigma);

lambda = 1e-1;
x_hat = f.ell2(H, Y, lambda);

% Show results
figure; hold on
subplot(221)
f.show_image(x)
title('Ground truth x')

subplot(222)
f.show_image(abs(x - x_hat))
title('Difference, |x - x\_hat|')

subplot(223)
f.show_image(y)
title('Observations y')

subplot(224)
f.show_image(x_hat)
title('Estimation x\_hat')

linkaxes
