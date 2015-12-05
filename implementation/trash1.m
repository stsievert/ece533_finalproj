im=double(image);
h=fspecial('gaussian',64,4);
figure;
imout=im;
for i=1:10
   display(i);
   imoutBig=padarray(imout,[size(im,1)/2,size(im,1)/2],'circular');
   hBig=padarray(h,[size(im,1)/2,size(im,2)/2],'circular');
   imout=conv2(imoutBig,hBig,'same');
   s1=size(image,1);
   s2=size(image,1);
   imout=imout(s1/2:end-s1/2-1,s2/2:end-s2/2)-1;
   clf;
   imagesc(imout);colormap gray;
   im=imout;
   display(size(imout));
   
   pause(1);
end
