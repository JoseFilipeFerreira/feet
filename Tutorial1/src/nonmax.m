function Z = nonMax(img, D)

    [rows, columns, ~] = size(img);
    Z = zeros(rows,columns);
    angle = D * 180 / pi;
    angle(angle < 0) = angle(angle < 0) + 180;
    
    for i = 2:(columns-1)
        for j = 2:(rows-1)
                q = 0;
                r = 0;
      
                %angle 0
                if ((0 <= angle(i,j) && angle(i,j) < 22.5) || (157.5 <= angle(i,j) && angle(i,j) <= 180))
                    q = img(i, j+1);
                    r = img(i, j-1);
                %angle 45
                elseif (22.5 <= angle(i,j) && angle(i,j) < 67.5)
                    q = img(i+1, j-1);
                    r = img(i-1, j+1);
                %angle 90
                elseif (67.5 <= angle(i,j) && angle(i,j) < 112.5)
                    q = img(i+1, j);
                    r = img(i-1, j);
                %angle 135
                elseif (112.5 <= angle(i,j) && angle(i,j) < 157.5)
                    q = img(i-1, j-1);
                    r = img(i+1, j+1);
                end

                if (img(i,j) >= q && img(i,j) >= r)
                    Z(i,j) = img(i,j);
                else
                    Z(i,j) = 0;
                end
        end
    end
end