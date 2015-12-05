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
gk=g;



f1=figure;f2=figure;
fk=f;
%This updation of gk is correct 
for i=1:RLucyIterations
    gk=fns.RLucyfnG(gk,fk,c);
    fk=conv2(f,gk,'same');
    figure(f1);imagesc(fk);colorbar ;colormap gray;
    title(num2str(mean(sum((fk(:)-f(:)).^2))));
    figure(f2);imagesc(conv2(f,gk,'same')-conv2(f,g,'same'));colorbar ;colormap gray;
    title('impovement in gk convolved with fk');
end

for i=1:RLucyIterations
    
end
