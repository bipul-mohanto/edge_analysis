function ctr = PixelCounter(bw)
%I = imread('rice.png');
%I = I(4:15, 18:29);
%figure; imshow(I);
%bw = im2bw(I)
%figure; imshow(bw);
ctr= 0;
[rows columns] = size(bw);
for i = 1 : rows
    for j = 1 : columns 
        if bw(i,j) == 1
            ctr = ctr + 1;
        end 
    end
end
%fprintf('1 are: %d \n', ctr);
%disp(ctr);
end