clc;close all;clear all;
a=rgb2gray(im2double(imread('in1.png')));
Actout=im2double(imread('out1.png'));
[m,n] = size(a);
K=[1 1];kx = K(1); ky = K(2);

op = [1 2 1;0 0 0;-1 -2 -1]./8; % Sobel approximation to derivative
x_mask = op'; % gradient in the X direction
y_mask = op;
scale = 4; % for calculating the automatic threshold
% compute the gradient in x and y direction
bx = imfilter(a,x_mask,'replicate');
by = imfilter(a,y_mask,'replicate');
    
% compute the magnitude
b = kx*bx.*bx + ky*by.*by;
    
% Determine cutoff based on RMS estimate of noise
% Mean of the magnitude squared image is a
% value that's roughly proportional to SNR
cutoff = scale*mean2(b);
e = b > cutoff;
figure;subplot(1,2,1);imshow(e);title('Sobel operator Output');
subplot(1,2,2);imshow(Actout);title('Expected Ideal Output');
% MSE
[m,n]=size(a);
emse=0;
for i2=1:m
for i3=1:n
    emse=emse+abs(double(Actout(i2,i3))-double(e(i2,i3)));
end
end
emse=emse/(m*n)
psnr=-10*log(emse/(max(double(Actout(:)))))
% create handle to the MSE_TEST function, that
% calculates MSE
h1 = @(x) NMSE(x, a , Actout);
h2 = @(x) thiscons(x);
% Setting the Genetic Algorithms tolerance for
% minimum change in fitness function before
% terminating algorithm to 1e-8 and displaying
% each iteration's results.

ga_opts = gaoptimset('display','iter');
ga_opts = gaoptimset(ga_opts, 'StallGenLimit', 100, 'Generations', 10);


% running the genetic algorithm with desired options
[x, err_ga] = ga(h1, 9,[],[],[],[],[],[],h2, ga_opts);

% 
op = reshape(x(1:9),3,3)./8;
x_mask = op'; % gradient in the X direction
y_mask = op;
scale = 4; % for calculating the automatic threshold
% compute the gradient in x and y direction
bx = imfilter(a,x_mask,'replicate');
by = imfilter(a,y_mask,'replicate');
    
% compute the magnitude
b = kx*bx.*bx + ky*by.*by;
    
cutoff = scale*mean2(b);
e = b > cutoff;
figure;subplot(1,2,1);imshow(e);title('GA opttimised operator Output');
subplot(1,2,2);imshow(Actout);title('Expected Ideal Output');
% MSE
[m,n]=size(a);
emse=0;
for i2=1:m
for i3=1:n
    emse=emse+abs(double(Actout(i2,i3))-double(e(i2,i3)));
end
end
emse=emse/(m*n)
psnr=-10*log(emse/(max(double(Actout(:)))))
