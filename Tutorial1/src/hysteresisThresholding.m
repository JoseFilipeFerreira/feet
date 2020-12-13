function img = hysteresisThresholding(img, weak, strong)

    [rows, columns, ~] = size(img);
    
    for i = (1: columns-1)
        for j = (1: rows-1)
            if (img(i,j) == weak)
                if ((img(i+1,j-1) == strong) || (img(i+1,j) == strong) || (img(i+1,j+1) == strong) || (img(i,j-1) == strong) || (img(i,j+1) == strong) || (img(i-1,j-1) == strong) || (img(i-1,j) == strong) || (img(i-1,j+1) == strong)) 
                    img(i,j) = strong;
                else
                    img(i,j) = 0;
                end
            end
        end
    end
end