function []=iterative_BC()
    fns=functions_();
    blur_sigma=8;
    [x, X, h, H, y, Y, n] = fns.input_and_observations(blur_sigma);
    
    % change of notation for direct correlation with the 1988 Paper
    % C is the only info available , 
    % f,F - reconstructed image, 
    % g,G - reconstructed filter
    C=Y; 
    
end