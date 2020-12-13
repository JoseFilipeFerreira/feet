function [res, weakV, strongV] = doubleThreshold(img, lowThresholdRatio, highThresholdRatio)

    %lowThresholdRatio = 0.5;
    %highThresholdRatio = 0.75;
    
    %Calculate Threshholds
    highThreshold = 1 - (max(max(img)) * highThresholdRatio);
    lowThreshold = highThreshold - (highThreshold * lowThresholdRatio);

    
    [rows, columns, ~] = size(img);
    res = zeros(rows,columns);
    
    %Weak and Strong pixel color values
    weakV = 0.1; %0.9 25
    strongV = 1; %0 255
    
    %Get strong and weak pixels in the image
    strong = img >= highThreshold;
    weak = ((img <= highThreshold) & (img >= lowThreshold));
    
    %Create a new image with the strong and weak pixels calculated
    res(strong) = strongV;
    res(weak) =  weakV;
    
end