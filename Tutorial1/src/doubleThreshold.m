
% function [res, weak, strong] = doubleThreshold(img, lowThresholdRatio, highThresholdRatio)
% 
%     highThreshold = max(max(img)) * highThresholdRatio;
%     lowThreshold = highThreshold * lowThresholdRatio;
%     
%     [rows, columns, ~] = size(img);
%     res = zeros(rows,columns);
%     
%     weak = 25; %0.9 25
%     strong = 255; %0 255
%     
%     
%     [strong_i, strong_j] = find(img >= highThreshold);
%     %[zeros_i, zeros_j] = find(img < lowThreshold);
%     
%     [weak_i, weak_j] = find((img <= highThreshold) & (img >= lowThreshold));
%     
%     res(strong_i, strong_j) = strong;
%     res(weak_i, weak_j) = weak;
%     
% end

% function [res, weak, strong] = doubleThreshold(img, lowThresholdRatio, highThresholdRatio)
% 
% 
%     highThreshold = min(min(img)) + 1*highThresholdRatio;
%     lowThreshold = highThreshold / lowThresholdRatio;
%     
%     [rows, columns, ~] = size(img);
%     res = zeros(rows,columns);
%     
%     weakV = 0.9; %0.9 25
%     strongV = 0; %0 255
%     
%     
%     strong = img <= highThreshold;
%     %[zeros_i, zeros_j] = find(img < lowThreshold);
%     
%     weak = (img >= highThreshold) & (img <= lowThreshold);
%     
%     res(strong) = strongV;
%     res(weak) =  weakV;
%     
% end


function [res, weakV, strongV] = doubleThreshold(img, lowThresholdRatio, highThresholdRatio)

    lowThresholdRatio = 0.6;
    highThresholdRatio = 0.83;
    
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