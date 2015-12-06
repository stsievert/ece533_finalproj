%deconvolution using MATLAB's deconvblind
I = imread('cameraman.bmp');
if(size(I,3)==3),I=rgb2gray(I);end
I=double(I);
I=I/max(I(:));
h=fspecial('gaussian',11,20);
noiseVariance=0.0001;
blurredImage=imfilter(I,h);
blurredNoisyImage=imnoise(blurredImage,'gaussian',0,noiseVariance);

WT=zeros(size(I));
WT(size(h,1)/2:end-size(h,1)/2+1,size(h,2)/2:end-size(h,2)/2+1)=1;
hest=ones(size(h));

NumIterations=20;
dampingFactor=10*sqrt(noiseVariance);
[J,P]=deconvblind(blurredNoisyImage,hest,NumIterations,dampingFactor,WT);

figure;
subplot(231);imagesc(I);colormap gray;title('original image');
subplot(232);imagesc(blurredNoisyImage);colormap gray;title('blurred Noisy image');
subplot(233);imagesc(h);colormap gray;title('true h');

subplot(234);imagesc(I);colormap gray;title('original image');
subplot(235);imagesc(J);colormap gray;title('recoverd image');
subplot(236);imagesc(h);colormap gray;title('recoverd h');