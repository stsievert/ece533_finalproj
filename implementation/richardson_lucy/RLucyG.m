blur_sigma=2;
fns=functions_();
[x, X, h, H, y, Y, n] = fns.input_and_observations(blur_sigma);
figure;fns.show(x,y,'original and blurred');

% variables - to be changed during experiments
blindSigma=4;
RLucyIterations=10;
TotalIterations=200;

c=y;
%initial estimates
f=y;g=fspecial('gaussian',size(x,1),blindSigma);


f1=figure;
fk=f;gk=g;
f2=figure;
for j=1:TotalIterations
    %This updation of gk is correct 
    figure(f2);
    fns.show(f,fk,'original and reconstructed Image');pause(1);
    figure(f1);
    for i=1:RLucyIterations
        gk=fns.RLucyfnG(gk,fk,c);
        fk=conv2(f,gk,'same');
        subplot(2,2,1);imagesc(fk);colorbar ;colormap gray;
        title([num2str(mean(sum((fk(:)-f(:)).^2))) 'reconstructed image']);
        subplot(2,2,2);imagesc(conv2(f,gk,'same')-conv2(f,g,'same'));colorbar ;colormap gray;
        title('impovement in gk convolved with fk G');
    end

    for i=1:RLucyIterations
        fk=fns.RLucyfnF(fk,gk,c);
        fk=conv2(fk,gk,'same');
        %error is diverging in this case !!
        subplot(2,2,3);imagesc(fk);colorbar;colormap gray;
        title(num2str(mean(sum((fk(:)-f(:)).^2))));
        subplot(2,2,4);imagesc(conv2(fk,gk,'same')-conv2(f,g,'same'));colorbar ;colormap gray;
        title('impovement in gk convolved with fk F');
    end

end
