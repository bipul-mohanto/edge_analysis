function [img_smooth] = find_edges(img,step_size,sigma)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%function to apply 2D gaussian filter to images
% hsize: filter step-size
% sigma: sigma value for gaussian function
if nargin < 1
error('the function requires atleast one argument: image');
end
if nargin < 2
step_size = [5 5]; %default value
sigma = 0.3 ;        %default value
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%subplot(1,2,1),imshow(img),title('original image')

img = double(img);
[r c]= size(img);
img_smooth = zeros(size(img));
G = fspecial('log',step_size,sigma);
min_r = floor(size(G,1)/2);
min_c = floor(size(G,2)/2);
max_r = r - min_r;
max_c = c - min_c;

for i = (1+min_r):max_r
    for j = (1+min_c):max_c
        row_min = i - min_r;
        row_max = i + min_r;
        col_min = j - min_c;
        col_max = j + min_c;
        template = double(img(row_min:row_max, col_min : col_max));
        img_smooth(i,j) = sum(sum(G .* template));
    end
end

% img_smooth = uint8(img_smooth);
% subplot(1,2,2),imshow(img_smooth),title('filtered image')
% figure, imagesc(img_smooth);colormap(gray); 
img_smooth = im2bw(img_smooth);
figure, imshow(img_smooth);
title ('Laplacian and Gaussian');

end