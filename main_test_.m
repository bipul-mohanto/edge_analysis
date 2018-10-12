clear all
close all
clc
warning ('off')
%% Ground truth
img = imread('6046o.jpg');
img_gray = rgb2gray(img); % gray image
img_ground_truth = imread('6046.png');
img_ground_truth = im2bw(img_ground_truth);
cc = img_ground_truth;
%cc = img_ground_truth(11:54, 112:171);

%% Detected part, can be imported from edges functions
%% Canny
% img_edge = edge(img_gray, 'canny');
% dd = img_edge;
%% Laplacian of Gaussian
% hsize = [7 7]; %only runs for odd integer step_size
% sigma = 0.3;
% img_gray = double(img_gray);
% dd = find_edges(img_gray,hsize,sigma); 

%% Kirsch
% dd = kirschedge(img_gray);

%% Robinson Edge Detection
% dd = robinsonedge(img_gray);

%% Genetic algorithm with sobel below
%% Sobel Algorithm
a = img_gray;
Actout=cc;
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
% figure, imshow(e); 
% title('Sobel operator Output');

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

%% Genetic Algorithm
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
f = b > cutoff;
%figure, imagesc(uint8(f));colormap(gray); 
figure, imshow(f);
title('GA opttimised operator Output');
dd = im2bw(f);

%% True pixel, false pixel, true negative 
[m n] = size(cc);
true_pixel = zeros(m,n);
false_pixel = zeros(m,n);
false_negative = zeros(m,n);
for i = 1:m
    for j = 1:n
        if (dd(i,j) == cc(i,j))
            true_pixel(i,j) = dd(i,j);                    
        elseif (dd(i,j)~= cc(i,j))
            false_pixel(i,j) = dd(i,j);
        end
    end
end
for i = 1:m
    for j = 1:n
        if (cc(i,j)==1 && dd(i,j)==0)
            false_negative(i,j) = cc(i,j);
        end
    end
end
% subplot(2,3,1); imshow(cc); title('Gt');
% subplot(2,3,2); imshow(dd); title('Dc');
figure(2), imshow(true_pixel, 'ColorMap',[1 1 1;0 1 0]); 
title('True Pixel (TP)');
hold on
figure(1),imshow(false_pixel, 'ColorMap',[1 1 1;1 0 0]); 
title('False Pixel (FP)');
hold on
figure(3),imshow(false_negative, 'ColorMap',[1 1 1;0 0 1]); 
title('False Negative (FN)');
hold on

%% create RGB channels for all-white image
% double to logical conversation
true_pixel = logical(true_pixel);
false_pixel = logical(false_pixel);
false_negative = logical(false_negative);

% creating three channels
r_channel = ones(size(true_pixel));
g_channel = ones(size(true_pixel));
b_channel = ones(size(true_pixel));

% leave pixels in true_pixel image GREEN
r_channel(true_pixel) = 0;
b_channel(true_pixel) = 0;

% leave pizels in false_pixel image RED
g_channel(false_pixel) = 0;
b_channel(false_pixel) = 0;

% leave pixels in false_negative image BLUE
r_channel(false_negative) = 0;
g_channel(false_negative) = 0;

% merge all three into RBG image
merged_image = cat(3, r_channel, g_channel, b_channel);
figure(4), imshow(merged_image);
title('merged image');

%% 1 pixel counter
FP = PixelCounter(false_pixel);
fprintf('False Pixels are: %d \n',FP);

TP = PixelCounter(true_pixel);
fprintf('True Pixels are: %d \n',TP);

FN = PixelCounter(false_negative);
fprintf('False Negatives are: %d \n',FN);

%% 0 pixel counter
TN = ((m*n)-(FP+TP+FN));
fprintf('True Negatives are: %d \n',TN);

%% bar plotting three values with three color

x = [TP*(100/(m*n)), FP*(100/(m*n)), FN*(100/(m*n)), TN*(100/(m*n))];
h = figure;
a = axes('parent',h);
hold (a, 'on');
colors = {'g','r','b', 'm'};
somenames = {'TP'; 'FP'; 'FN'; 'TN'};
for i = 1:numel(x)
    b = bar(i, x(i), 0.50, 'stacked', 'parent', a , 'facecolor', colors{i});
end
set(a,'XTick',1:4);
set(a,'XTickLabel',somenames);
ylabel('number of pixels (%)');
%% List of Error measures only statistics

% 1. complement Dice Measure
dice = 1-(2*TP/(2*TP+FN+FP));
fprintf('complemented Dice measure: %.3f \n',dice);

% 2. complement performance measure (P_m)
P_m = 1-(TP/(TP+FP+FN));
fprintf('complement performance measure: %.3f \n',P_m);

% 3. Complemented Absolute Grading 
A_g = 1-(TP/sqrt((TP+FN)*(TP+FP)));
fprintf('Complemented Absolute Grading: %.3f \n',A_g);

% 4. Complemented Segmentation success Ratio
ssr = 1-(TP.^2/((TP+FN)*(TP+FP)));
fprintf('Complemented Segmentation Success Ratio: %.3f \n',ssr);

% 5. Localization error
I = TP+TN+FP+FN;
P_e = (FP+FN)/abs(I); 
fprintf('Localization error: %.3f \n',P_e);


% 6. Misclassification error
m_error =  1-((TP+TN)/(TN+FN+TP+FP));
fprintf('Misclassification error: %.3f \n',m_error);


% 7. complemented fai measure
TPR = TP/(TP+FN);
fai = 1 -((TPR*TN)/(TN+FP));
fprintf('complemented fai measure: %.3f \n',fai);

% 8. Complemented F_alpha measure
PREC = TP/(TP+FP);
for alpha = 0.1:0.4:1
F_alpha = 1-(PREC*TPR)/((alpha*TPR)+((1-alpha)*PREC));
fprintf('complemented f_alpha measure: %.3f \n',F_alpha);
end 


%% complemented x^2 measure
% FPR = FP/(FP+TN);
% x_square = 1-(((TPR-TP-FP)/(1-TP-FP))*((TP+FP+FPR)/(TP+FP)));
% fprintf('complemented x square measure: %.3f \n',x_square);

close all



