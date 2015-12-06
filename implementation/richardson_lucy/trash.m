blur_sigma=2;
fns=functions_();
[x, X, h, H, y, Y, n] = fns.input_and_observations(blur_sigma);
figure;fns.show(x,y,'original and blurred');

% variables - to be changed during experiments
blindSigma=4;RLucyIterations=40;TotalIterations=4;

%initial estimates
c=y;f=y;g=fspecial('gaussian',size(x,1),blindSigma);
g=ones(size(g));

f1=figure;f2=figure;
fk=f;gk=g;gkTemp=0;errorTemp=Inf;

for j=1:TotalIterations   
    figure(f1);
    for i=1:RLucyIterations
        gkTemp=fns.RLucyfnG(gk,fk,c);
        if(sum((gk(:)-h(:)).^2)<Inf)
           gk=gkTemp; errorTemp=sum((gk(:)-h(:)).^2);
        end
          fprintf('%f\n',(sum((gk(:)-h(:)).^2)));
        
        [fk,Fk]=fns.weiner(fft2(gk),fft2(fk));
        fprintf('gk error %f\n',(sum((gk(:)-h(:)).^2)));
        fprintf('fk error %f\n',(sum((fk(:)-x(:)).^2)));
        subplot(131);imagesc(y);colormap gray;colorbar;title('blurred image');
        subplot(132);imagesc(fk);colormap gray;colorbar;title('reconstructed image');
        subplot(133);imagesc(x-fk);colormap gray;colorbar;title('x-fk');
        pause(1);
    end
   
end
