blur_sigma=2;
fns=functions_();
[x, X, h, H, y, Y, n] = fns.input_and_observations(blur_sigma);
figure;fns.show(x,y,'original and blurred');

% variables - to be changed during experiments
blindSigma=4;
RLucyIterations=40;
TotalIterations=4;

c=y;
%initial estimates
f=y;
g=fspecial('gaussian',size(x,1),blindSigma);
g=ones(size(g));


f1=figure;
fk=f;gk=g;
f2=figure;
for j=1:TotalIterations
    %This updation of gk is correct 
    figure(f2);
   % fns.show(f,fk,'original and reconstructed Image');pause(1);
    figure(f1);
    for i=1:RLucyIterations
        gk=fns.RLucyfnG(gk,fk,c);
        fprintf('%f\n',(sum((gk(:)-h(:)).^2)));
        %fk=conv2(f,gk,'same');
%         subplot(1,2,1);imagesc(fk);colorbar ;colormap gray;
%         title([num2str(mean(sum((fk(:)-f(:)).^2))) 'reconstructed image']);
%         subplot(1,2,2);imagesc(conv2(f,gk,'same')-conv2(f,g,'same'));colorbar ;colormap gray;
%         title('impovement in gk convolved with fk G');
%         pause(1);
    end
% 
%     for i=1:RLucyIterations
%         fk=fns.RLucyfnG(fk,gk,c);
%         fk=fk/max(fk(:));
%         fprintf('image errors = %f\n',(sum((fk(:)-f(:)).^2)));
% %         fk=conv2(fk,gk,'same');
% %         %error is diverging in this case !!
% %         subplot(2,2,3);imagesc(fk);colorbar;colormap gray;
% %         title(num2str(mean(sum((fk(:)-f(:)).^2))));
% %         subplot(2,2,4);imagesc(conv2(fk,gk,'same')-conv2(f,g,'same'));colorbar ;colormap gray;
% %         title('impovement in gk convolved with fk F');
%     end
    subplot(221);imagesc(fk);colormap gray;colorbar;title('reconstructed')
    subplot(222);imagesc(y);colormap gray;colorbar;title('original blurred image')
    subplot(223);imagesc(y-fk);colormap gray;colorbar;title('fk-y')
    pause(10);
end
