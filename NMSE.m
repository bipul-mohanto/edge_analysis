function emse = NMSE( x, a, Actout)
K=[1 1];kx = K(1); ky = K(2);

op = reshape(x,3,3)./8; % operator approximation to derivative
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
% MSE
[m,n]=size(a);
emse=0;
for i2=1:m
for i3=1:n
    emse=emse+abs(double(Actout(i2,i3))-double(e(i2,i3)));
end
end
emse=emse/(m*n);
psnr=-10*log(emse/(max(double(Actout(:)))));