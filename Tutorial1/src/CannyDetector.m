
%Inputs
%img = imread('C:\Users\lpfer\Desktop\VC\Tutorial1\images\lena.jpg');
%img = imread('C:\Users\lpfer\Desktop\VC\Tutorial1\images\baboon.png', 'png');
%gaussianImg = imnoise(img,'gaussian',0.01,0.001);

img = imread('C:\Users\lpfer\Desktop\VC\Tutorial1\images\flower_gaussian_0.2.png');
bwImg = im2double(im2gray(gaussianImg));
%figure,imshow(bwImg);
sigma = 1.2; %Larger sigma, better smoothing, more blur, less detailed image
x = 13; y = 13;

[rows, columns, ~] = size(bwImg);
newImg = zeros(rows+2*y, columns+2*x);
for i = (x+1):(columns+x-1)
    for j = (y+1):(rows+y-1)
        newImg(i,j) = bwImg(i-x+1,j-y+1);
    end
end
%figure,imshow(newImg);
%imwrite(newImg,'C:\Users\lpfer\Desktop\VC\Tutorial1\images2\borderedImage.png');

%Gaussian_smoothing.m
gsImg = gaussianSmoothing(newImg,sigma,x,y);
%figure, imshow(gsImg);
% imwrite(gsImg,'C:\Users\lpfer\Desktop\VC\Tutorial1\images2\flowerSize33.png');
% 
% gsImg = gaussianSmoothing(newImg,sigma,13,13);
% figure, imshow(gsImg);
% imwrite(gsImg,'C:\Users\lpfer\Desktop\VC\Tutorial1\images2\flowerSize13.png');

%gradient.m
[G,theta] = gradient(gsImg);
% figure, imshow(G);
% imwrite(G,'C:\Users\lpfer\Desktop\VC\Tutorial1\images2\flowerGrad.png');

%nonmax
Z = nonMax(G,theta);
%figure, imshow(Z);
%imwrite(Z,'C:\Users\lpfer\Desktop\VC\Tutorial1\images2\flowerNonMax.png');

%double_threshold
%[res, weak, strong] = doubleThreshold(Z, 0.05, 0.09);
[res, weak, strong] = doubleThreshold(Z, 0.3, 0.72);
%figure, imshow(res);
%imwrite(res,'C:\Users\lpfer\Desktop\VC\Tutorial1\images2\flowerDoubleThreshold.png');

%hysteresis_thresholding
finalImg = hysteresisThresholding(res, weak, strong);
%figure, imshow(finalImg);
imwrite(finalImg,'C:\Users\lpfer\Desktop\VC\Tutorial1\images2\flowerHysteresis.png');
lastImg = finalImg((x+1):(columns+x), (y+1):(rows+y));
%figure,imshow(lastImg);
imwrite(lastImg,'C:\Users\lpfer\Desktop\VC\Tutorial1\images2\flowerFinal.png');

    
%edge strength image BEFORE nonmax suppression
imwrite(G,'C:\Users\lpfer\Desktop\VC\Tutorial1\images2\flower_edge_canny_13_1.2.png');

%edge strength image AFTER nonmax suppression
imwrite(Z,'C:\Users\lpfer\Desktop\VC\Tutorial1\images2\flower_edge_canny_nonmax_13_1.2.png');

%edge strength image AFTER hysteresis thresholding
imwrite(lastImg,'C:\Users\lpfer\Desktop\VC\Tutorial1\images2\flower_edge_canny_hysteresis_13_1.2.png'); 

Sobel = edge(bwImg,'Sobel');
figure, imshow(Sobel);
imwrite(Sobel,'C:\Users\lpfer\Desktop\VC\Tutorial1\images2\flower_Matlabedge_Sobel.png');

Prewitt = edge(bwImg,'Prewitt');
figure, imshow(Prewitt);
imwrite(Prewitt,'C:\Users\lpfer\Desktop\VC\Tutorial1\images2\flower_Matlabedge_Prewitt.png');

Roberts = edge(bwImg,'Roberts');
figure, imshow(Roberts);
imwrite(Roberts,'C:\Users\lpfer\Desktop\VC\Tutorial1\images2\flower_Matlabedge_Roberts.png');

Canny = edge(bwImg,'Canny',([0.001 0.99]),1.2);
figure, imshow(Canny);
imwrite(Canny,'C:\Users\lpfer\Desktop\VC\Tutorial1\images2\flower_Matlabedge_Canny_1.2.png');


    
    