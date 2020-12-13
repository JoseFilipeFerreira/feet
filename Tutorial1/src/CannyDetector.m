
%Inputs
img = imread('C:\Users\lpfer\Desktop\VC\Tutorial1\images\lena.jpg');
%img = imread('C:\Users\lpfer\Desktop\VC\Tutorial1\images\baboon.png', 'png');
gaussianImg = imnoise(img,'gaussian',0.01,0.001);
bwImg = im2double(im2gray(gaussianImg));
%figure,imshow(bwImg);
sigma = 1; %Larger sigma, better smoothing, more blur, less detailed image
x = 5; y = 5;

[rows, columns, ~] = size(bwImg);
newImg = zeros(rows+2*y, columns+2*x);
for i = (x+1):(columns+x-1)
    for j = (y+1):(rows+y-1)
        newImg(i,j) = bwImg(i-x+1,j-y+1);
    end
end
%figure,imshow(newImg);

%Gaussian_smoothing.m
gsImg = gaussianSmoothing(newImg,sigma,x,y);
%figure, imshow(gsImg);

%gradient.m
[G,theta] = gradient(gsImg);
%figure, imshow(theta);
%figure, imshow(G);

%nonmax
Z = nonMax(G,theta);
%figure, imshow(Z);

%double_threshold
%[res, weak, strong] = doubleThreshold(Z, 0.05, 0.09);
[res, weak, strong] = doubleThreshold(Z, 0.05, 0.09);
figure, imshow(res);

%hysteresis_thresholding
finalImg = hysteresisThresholding(res, weak, strong);
figure, imshow(finalImg);
    
%edge strength image BEFORE nonmax suppression
imwrite(G,'C:\Users\lpfer\Desktop\VC\Tutorial1\images2\OriginalName_edge_canny_filtersize_variance.png');

%edge strength image AFTER nonmax suppression
imwrite(Z,'C:\Users\lpfer\Desktop\VC\Tutorial1\images2\OriginalName_edge_canny_nonmax_filtersize_variance.png');

%edge strength image AFTER hysteresis thresholding
imwrite(finalImg,'C:\Users\lpfer\Desktop\VC\Tutorial1\images2\OriginalName_edge_canny_hysteresis_filtersize_variance.png'); 

% figure, imshow(edge(bwImg,'Sobel'));
% figure, imshow(edge(bwImg,'Prewitt'));
% figure, imshow(edge(bwImg,'Roberts'));
% figure, imshow(edge(bwImg,'Canny'));

% lastImg = finalImg((x+1):(columns+x), (y+1):(rows+y));
% figure,imshow(lastImg);
    
    