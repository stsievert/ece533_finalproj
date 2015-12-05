function f = functions_()
    f.show_image = @show_image;
    f.ell2 = @ell2;
    f.input_and_observations = @input_and_observations;
    f.reconstruct1=@reconstruct1;
    f.conserve_energy=@conserve_energy;
    f.show=@show;
    f.RLucyfnG=@RLucyfnG;
    f.RLucyfnF=@RLucyfnF;
end

function [x, X, h, H, y, Y, n] = input_and_observations(blur_sigma)
    % :param blur_sigma: Characterizes the gaussian standard deviation of the
    %                    space domain kernel function for convolution. Larger
    %                    blur_sigma means more a blurryier image
    % :returns x, X: The space and Fourier representations (resepecitvely)
    %                of the input
    % :returns h, H: The space and Fourier representations (resepecitvely)
    %                of the convolution matrix
    % :returns y, Y: The space and Fourier representations (resepecitvely)
    %                of the observations
    
    % Adding - uigetfile to open a file along with some memory of last
    % location
    x = ( imread('cameraman64.png'));
    if(size(x,3)==3),x=rgb2gray(x);x=double(x);end % conversion to grayscale
    if(size(x,1)~=size(x,2)),x=x(1:min(size(x,1),size(x,2)),1:min(size(x,1),size(x,2))); end %considering only square images
    x = x(:, :, 1) / max(x(:)); % assume gray scale image
    n = size(x, 1);
    X = fft2(x);

    h = fspecial('gaussian', n, blur_sigma);
    h = h / sum(h(:));
    H = fft2(h);

    Y = X .* abs(H);
    y = ifft2(Y);
    y = abs(y);
    y=y/max(y(:));
end

function x_hat = ell2(H, Y, lambda)
    % Performs
    %   \widehat{x} = \arg \min_x ||y - hx||_2^2 + \lambda ||x||_2^2
    %
    % :param H: The 2D fourier transform representation of h
    % :param Y: The 2D Fourier transform representation of y
    % :param lambda: The regularization scaling parameter. How strongly do we
    %                want to emphasize the importance of each term?
    % :returns x_hat: The estimate of x from observations y
    n = size(H, 1);
    G = eye(n);
    W = conj(H') ./ (abs(H).^2 + lambda*G.^2 + 1e-9);
    x_hat = abs(ifft2(W .* Y));
    x_hat = ifftshift(x_hat);
end

function show_image(x)
    % A little helper function to show an image with the right colormap and show
    % a colorbar
    imagesc(x);
    colormap gray
    colorbar
end

function Fnew=reconstruct1(Fold,G,C,BETA,noise_level)
%     Inputs - 
%     Fold - old fft transform of reconsrtucted image of last step
%     G- fft transform of filter of last step
%     C- fft transform of the output image
%     Beta - control for estimation of Fnew- domain=[0,1]
%     noise_level - noise level in the originall image - can be an estimated value
%     Output - 
%     Fnew - new estimate of F
    
    %initialisations
    [s1,s2]=size(Fold); Fnew(1:s1,1:s2)=0;
    epsilon=1e-4;
    for i=1:s1
        for j=1:s2
            if(abs(C(i,j)<noise_level))
                Fnew(i,j)=Fold(i,j);
            elseif(abs(G(i,j))>=abs(C(i,j)))
                Fnew(i,j)=(1-BETA)*Fold(i,j)+BETA*C(i,j)/(G(i,j)+epsilon);
            elseif(abs(G(i,j))<abs(C(i,j)))
                temp=(1-BETA)/(Fold(i,j)+epsilon)+BETA*G(i,j)/(C(i,j)+epsilon);
                Fnew(i,j)=1/(temp+epsilon);
            end
        end
    end
    return;
end

function fnew=conserve_energy(f)
    % conserves the energy in image f. For all values in f <0, ,mean of 
    % absolute values is added to each pixel in f.
    % process is repeated till no pixel in fnew is <lower_limit . Because 
    % the minimum pixel value will never be greater than 0
    
    % This functionality assumes that the number of pixels with negative
    % values are less. 
    % Output - fnew - contains no element with value<0
    
    % Initialisations 
    [s1,s2]=size(f);fnew(1:s1,1:s2)=0;f=double(f);
    %lower_limit=-0.000001;
    lower_limit=0;
    iteration_number=1;num_iterations=5;
    FLAG=0;% 0 INDICATES THE LOOP WAS NOT ENTERED- output f itself
    while(min(f(:))<lower_limit&&iteration_number<num_iterations)
        FLAG=1;
        E=0;
         for i=1:s2
            for j=1:s2
                if(f(i,j)<lower_limit)
                    fnew(i,j)=0;E=E+abs(f(i,j));
                else
                   fnew(i,j)=f(i,j); 
                end
            end
        end
        fnew=fnew+E/(s1*s2);
        f=fnew;
        iteration_number=iteration_number+1;
    end
    fnew(fnew<0)=-lower_limit;
   if(FLAG==0),fnew=f;end;
       
end

function []=show(f1,f2,s1)
    clf;
     subplot(121);imagesc(f1);colormap gray;colorbar;
    subplot(122);imagesc(f2);colormap gray;colorbar;
      title(s1);
      pause(5);
end

function [gknew]=RLucyfnG(gkold,fkold,c)
    gknew=zeros(size(gkold));
    gknew=conv2(gkold,fkold,'same');%debug(gknew)
    gknew=c./gknew;%debug(gknew)
    gknew=conv2(gknew,fliplr(flipud(fkold)),'same');%debug(gknew)
    gknew=gknew.*gkold;%debug(gknew)
    gknew=gknew/sum(gknew(:));
%    pause(1);
end

function[fknew]=RLucyfnF(fkold,gkold,c)
    %figure;
    fknew=zeros(size(fkold));           debug(fknew);
    fknew=conv2(fkold,gkold,'same');    debug(fknew);
    fknew=c./fknew;                     debug(fknew);
    fknew=conv2(fknew,fliplr(flipud(gkold)),'same');    debug(fknew);
    fknew=fknew.*fkold;                 debug(fknew);
    fknew=fknew/max(fknew(:));          debug(fknew);
end

function []=debug(gknew)
    %imagesc(gknew);colormap gray;colorbar
end