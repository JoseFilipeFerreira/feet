function maxs = specialMax(img,n)

    [rows, columns, ~] = size(img);
    nMaxs = zeros(rows,n);
     
    for i = 1:rows
        for j = 1:columns
            for y = 1:n
                if img(i,j) > nMaxs(i,y)
                    for y2 = y:(n-1)
                        nMaxs(i,y2+1) = nMaxs(i,y2);
                    end
                    nMaxs(i,y) = img(i,j);
                    break
                end
            end     
        end
    end

    maxs = mean(nMaxs')';
end