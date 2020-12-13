function gsImg = gaussianSmoothing(img,sigma,x,y)
    newX = x-1; newY = y-1;
    [rows, columns, ~] = size(img);

    %Create kernel window with given inputs
    %w = fspecial('gaussian',x,sigma);
    w = ones(newX,newX);
    for x1 = (-newX/2):(newX/2)
        for y1 = (-newY/2):(newY/2)
            w(x1+(newX/2)+1,y1+(newY/2)+1) = (1/(2*pi*sigma^2))*exp(-(x1^2+y1^2)/(2*sigma^2));
        end
    end  

    %Normalize Window
    normW = w ./ sum(sum(w));
    
    %Smooth image with gaussian filter
    %gsImg = imfilter(img, normW, 'replicate'); %replicate circular symmetric 0
    
    gsImg = zeros(rows,columns);
    sizeK = floor(x/2);
    for i = (1+sizeK):(columns-sizeK)
        for j = (1+sizeK):(rows-sizeK)
            for u = -sizeK:sizeK
                for v = -sizeK:sizeK
                    gsImg(i,j) = gsImg(i,j) + normW(u+1+sizeK,v+1+sizeK) * img(i-u,j-v);
                end
            end
        end
    end
    
end

