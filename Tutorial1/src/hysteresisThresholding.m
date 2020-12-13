function img = hysteresisThresholding(img, weak, strong)

    [rows, columns, ~] = size(img);
    
    for i = (1: columns-1)
        for j = (1: rows-1)
            if (img(i,j) == weak)
                if ((img(i+1,j-1) == strong) || (img(i+1,j) == strong) || (img(i+1,j+1) == strong) || (img(i,j-1) == strong) || (img(i,j+1) == strong) || (img(i-1,j-1) == strong) || (img(i-1,j) == strong) || (img(i-1,j+1) == strong)) 
                    img(i,j) = strong;
                else
                    img(i,j) = weak;
                end
            end
        end
    end
    
    for i = (1: columns-1)
        for j = (1: rows-1)
            if (img(columns-i,rows-j) == weak)
                if ((img(columns-i+1,rows-j-1) == strong) || (img(columns-i+1,rows-j) == strong) || (img(columns-i+1,rows-j+1) == strong) || (img(columns-i,rows-j-1) == strong) || (img(columns-i,rows-j+1) == strong) || (img(columns-i-1,rows-j-1) == strong) || (img(columns-i-1,rows-j) == strong) || (img(columns-i-1,rows-j+1) == strong)) 
                    img(columns-i,rows-j) = strong;
                else
                    img(columns-i,rows-j) = 0;
                end
            end
        end
    end
end