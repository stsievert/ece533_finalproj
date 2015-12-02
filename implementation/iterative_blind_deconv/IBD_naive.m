 %Naive Approach - no exception handling - just imposing image constraints
    fns=functions_();
    blur_sigma=20;
    [x, X, h, H, y, Y, n] = fns.input_and_observations(blur_sigma);
    
    figure;
    subplot(121);imagesc(y);colormap gray;title('filtered');
    subplot(122);imagesc(x);colormap gray;title('orignial');
    
    % change of notation for direct correlation with the 1988 Paper
    % C is the only info available , 
    % f,F - reconstructed image, 
    % g,G - reconstructed filter
    C=Y; epsilon=1e-4;
    
    %initial estimate of f cannot be uniform- zeros in Freq domain F
    f=double(randn(size(x,1)));%f here can be negative
    f=fns.conserve_energy(f);%f should be non negative
   
    %Step1
        F=fft2(f);
    %Step2
        G=C./F;     
    %Step3 
        g=real(ifft2(G));
    %Step4
        g=fns.conserve_energy(g);
    %Step5
        G=fft2(g);
    %Step6
        F=C./G;
    %Step7
        f=real(ifft2(F));
    %Step8
        f=fns.conserve_energy(f);
        
    MAX_ITERRATIONS=20;
    figure;
    for i=1:MAX_ITERRATIONS
       %Step1
        F=fft2(f);
    %Step2
        G=C./F;     
    %Step3 
        g=real(ifft2(G));
    %Step4
        g=fns.conserve_energy(g);
    %Step5
        G=fft2(g);
    %Step6
        F=C./G;% this is inverse filtering that's why naive
    %Step7
        f=real(ifft2(F));
    %Step8
        f=fns.conserve_energy(f);
        subplot(121);imagesc(x);colormap gray;title('original image');
        subplot(122);imagesc(f);colormap gray;title(['Reconstructed image ' num2str(i) ' iteration']);
        pause(1);
    end