%A FUNCTION TO DETECT EDGES IN THE IMAGE

function y=robinsonedge(x)
    
    %x=double(x);


    g1=[1,2,1;0,0,0;-1,-2,-1];
    g2=[2 1 0;1 0 -1;0 -1 -2];
    g3=[1 0 -1;2 0 -2;1 0 -1];
    g4=[0 -1 -2;1 0 -1;2 1 0];
    g5=[-1 -2 -1;0 0 0;1 2 1];
    g6=[-2 -1 0;-1 0 1;0 1 2];
    g7=[-1 0 1;-2 0 2;-1 0 1];
    g8=[0 1 2;-1 0 1;-2 -1 0];


    x1=imfilter(x,g1,'replicate');
    x2=imfilter(x,g2,'replicate');
    x3=imfilter(x,g3,'replicate');
    x4=imfilter(x,g4,'replicate');
    x5=imfilter(x,g5,'replicate');
    x6=imfilter(x,g6,'replicate');
    x7=imfilter(x,g7,'replicate');
    x8=imfilter(x,g8,'replicate');

    y1=max(x1,x2);
    y2=max(y1,x3);
    y3=max(y2,x4);
    y4=max(y3,x5);
    y5=max(y4,x6);
    y6=max(y5,x7);
    y7=max(y6,x8);
    y=y7;
    y = im2bw(y);
    %figure, imagesc(uint8(y));colormap(gray);
    figure, imshow(y);
    title ('Robinson Edge');
end