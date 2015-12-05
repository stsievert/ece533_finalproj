fns=functions_();
    blur_sigma=4;
    [x, X, h, H, y, Y, n] = fns.input_and_observations(blur_sigma);
     MAX_ITERATIONS=20;
    
    BETA=0.01;
    noiseLevel=0.00;
    noiseLevelEstimated=0.01;
    C=Y;
    f=y+noiseLevel*randn(size(y,1));
    f=fns.normalize(f);
    f=fns.conserve_energy(f);
    F=fft2(f);
    
    g=y+noiseLevel*randn(size(y,1));
    g=fns.normalize(g);
    g=fns.conserve_energy(g);
    G=fft2(g);
    
    figure;fns.show(abs(ifft2(F)),abs(ifft2(G)),'initial estimates of f and g');
    f=abs(ifft2(F));g=abs(ifft2(G));
    f=fns.conserve_energy(fns.normalize(f));
    g=fns.conserve_energy(fns.normalize(g));
    
% Initial estimagtes of f and g are correct now

for i=1:MAX_ITERATIONS
    %Step1
        F=fft2(f);
    %Step2 
        G=fns.reconstruct2(G,F,C,BETA,noiseLevelEstimated);
        figure;fns.show(abs(ifft2(C)),abs(ifft2(G.*F)),'C and G.*F');
    %Step3
        g=fns.conserve_energy(fns.normalize((ifft2(G))));
end
    
%     G2=fns.reconstruct2(G,F,C,BETA,noiseLevelEstimated);
%     F2=F./G2;
%     C2=F2.*G2;
%     fns.show(abs(ifft2(C)),abs(ifft2(C2)),'C and F.*G2');
%     pause(2);