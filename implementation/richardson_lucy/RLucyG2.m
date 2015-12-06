blur_sigma=2;
fns=functions_();
[x, X, h, H, y, Y, n] = fns.input_and_observations(blur_sigma);
figure;fns.show(x,y,'original and blurred');

% variables - to be changed during experiments
blindSigma=4;RLucyIterations=40;TotalIterations=4;


%initial estimates
c=y;f=y;%g=fspecial('gaussian',size(x,1),blindSigma);
g=ones(size(g));

f1=figure;f2=figure;
fk=f;gk=g;gkTemp=0;errorTemp=Inf;
for j=1:TotalIterations
    %This updation of gk is correct 
    figure(f2);
   % fns.show(f,fk,'original and reconstructed Image');pause(1);
    figure(f1);
    for i=1:RLucyIterations
        gkTemp=fns.RLucyfnG(gk,fk,c);
        if(sum((gk(:)-h(:)).^2)<Inf)
           gk=gkTemp; 
        end
        fprintf('%f\n',(sum((gk(:)-h(:)).^2)));
        
    end
    
    [fk,Fk]=fns.weiner(fft2(gk),fft2(fk));
    fk=fk/max(fk(:));
    subplot(221);imagesc(fk);colormap gray;colorbar;title('reconstructed')
    subplot(222);imagesc(y);colormap gray;colorbar;title('original blurred image')
    subplot(223);imagesc(y-fk);colormap gray;colorbar;title('fk-y')
    pause(10);
    % Here we are getting a good estimate of h as gk
    % now using weiner filter
    
end
