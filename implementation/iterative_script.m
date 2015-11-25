
    fns=functions_();
    blur_sigma=8;
    [x, X, h, H, y, Y, n] = fns.input_and_observations(blur_sigma);
    
    % change of notation for direct correlation with the 1988 Paper
    % C is the only info available , 
    % f,F - reconstructed image, 
    % g,G - reconstructed filter
    C=Y; 
    
    %initial estimates
    f=ones(size(x));F=fft2(f);
    G=C./F;
    
    g=real(ifft2(G));
    g(g<0)=0;
    G=fft2(g);
    
    %now G and F are initialised and we need to do iterations - function
    %used - reconstruct1 
