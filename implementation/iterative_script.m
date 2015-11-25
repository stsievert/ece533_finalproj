
    fns=functions_();
    blur_sigma=8;
    [x, X, h, H, y, Y, n] = fns.input_and_observations(blur_sigma);
    
    % change of notation for direct correlation with the 1988 Paper
    % C is the only info available , 
    % f,F - reconstructed image, 
    % g,G - reconstructed filter
    C=Y; 
    
    %initial estimates
%     f=randn(size(x,1));
    f=y+1e-10*randn(size(x,1));
    %f=y;
    f=double(f);F=fft2(f);
    G=C./F;
    
    g=real(ifft2(G));
    g(g<0)=0;% Error
    G=fft2(g);
    
    %now G and F are initialised and we need to do iterations - function
    %used - reconstruct1 
    iterations=200;BETA=0.9;iteration_number=1;
    noise_level=0.01;% no noise for now
    
    figure;% for plotting results
    while(iteration_number<=iterations)
        clf;
        subplot(121);imagesc(x);colormap gray;
        subplot(122);imagesc(f);colormap gray;
       pause(1);
        %step 6- finding F from G and C
        Fnew=fns.reconstruct1(G,F,C,BETA,noise_level);
        F=Fnew;
        %step 7 - finding f from F
        f=real(ifft2(F));
        
        %step 8 applying image constraints
        f=fns.conserve_energy(f);% not working as expected
        
        %step 1 - finding F from f
        F=fft2(f);
        
        %Step 2 - finding G from F and C
        G=fns.reconstruct1(F,G,C,BETA,noise_level);
        
        %Step 3 - finding g from G
        g=real(ifft2(G));
        
        %Step 4 - putting constraints on g
        g=fns.conserve_energy(g);
        
        %Step 5 - finding G from g
        G=fft2(g);
        iteration_number=iteration_number+1;
    end