function f = functions_()
    f.show_image = @show_image;
    f.ell2 = @ell2;
    f.input_and_observations = @input_and_observations;
end

function [x, X, h, H, y, Y, n] = input_and_observations(blur_sigma)
    % :param blur_sigma: Characterizes the gaussian standard deviation of the
    %                    space domain kernel function for convolution. Larger
    %                    blur_sigma means more a blurryier image
    % :returns x, X: The space and Fourier representations (resepecitvely)
    %                of the input
    % :returns h, H: The space and Fourier representations (resepecitvely)
    %                of the convolution matrix
    % :returns y, Y: The space and Fourier representations (resepecitvely)
    %                of the observations
    x = double( imread(['data/', 'cameraman.png']) );
    x = x(:, :, 1) / max(x(:)); % assume gray scale image
    n = size(x, 1);
    X = fft2(x);

    h = fspecial('gaussian', n, blur_sigma);
    h = h / sum(h(:));
    H = fft2(h);

    Y = X .* abs(H);
    y = ifft2(Y);
    y = abs(y);
end

function x_hat = ell2(H, Y, lambda)
    % Performs
    %   \widehat{x} = \arg \min_x ||y - hx||_2^2 + \lambda ||x||_2^2
    %
    % :param H: The 2D fourier transform representation of h
    % :param Y: The 2D Fourier transform representation of y
    % :param lambda: The regularization scaling parameter. How strongly do we
    %                want to emphasize the importance of each term?
    % :returns x_hat: The estimate of x from observations y
    n = size(H, 1);
    G = eye(n);
    W = conj(H') ./ (abs(H).^2 + lambda*G.^2 + 1e-9);
    x_hat = abs(ifft2(W .* Y));
    x_hat = ifftshift(x_hat);
end

function show_image(x)
    % A little helper function to show an image with the right colormap and show
    % a colorbar
    imagesc(x);
    colormap gray
    colorbar
end
