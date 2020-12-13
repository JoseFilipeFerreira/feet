function [res, weakV, strongV] = doubleThreshold(img, lowThresholdRatio, highThresholdRatio)
    
    %Calculate Threshholds
    highThreshold = 1 - (max(max(img)) * highThresholdRatio);
    lowThreshold = highThreshold - (highThreshold * lowThresholdRatio);

    
    [rows, columns, ~] = size(img);
    res = zeros(rows,columns);
    
    %Weak and Strong pixel color values
    weakV = 0.2; %0.2 25
    strongV = 1; %0 255
    
    %Get strong and weak pixels in the image
    strong = img >= highThreshold;
    weak = ((img <= highThreshold) & (img >= lowThreshold));
    
    %Create a new image with the strong and weak pixels calculated
    res(strong) = strongV;
    res(weak) =  weakV;
    
end