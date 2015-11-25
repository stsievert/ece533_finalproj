fns=functions_();
    blur_sigma=8;
    [x, X, h, H, y, Y, n] = fns.input_and_observations(blur_sigma);
    
    %C =F*G - only play with fft magnitude values and not phase values
    C=Y;
    f=ones(size(x,1));
    F=fft2(f);G=C./F;
    
    Fmag=abs(F);Cmag=abs(C);Gmag=abs(G);
    MAXITER=100;
    for i=1:MAXITER
        
    end