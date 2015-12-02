I=checkerboard(8);
PSF = fspecial('gaussian',7,10);
V = .0001;
BlurredNoisy = imnoise(imfilter(I,PSF),'gaussian',0,V);

WT = zeros(size(I));
WT(5:end-4,5:end-4) = 1;
INITPSF = ones(size(PSF));
figure;
for i=1:20
    [J P] = deconvblind(BlurredNoisy,INITPSF,i,10*sqrt(V),WT);
    subplot(231);imshow(BlurredNoisy);
    title('A = Blurred and Noisy');
    subplot(232);imshow(PSF,[]);
    title('True PSF');
    subplot(233);imshow(J);
    title('Deblurred Image');
    subplot(234);imshow(P,[]);
    title('Recovered PSF');
    subplot(235);imshow(J-I,[]);
    title('Error from ground truth');
    subplot(236);imshow(I,[]);
    title('ground truth');
    
    display(i);
    pause(1);
end
