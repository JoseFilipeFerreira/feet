function [G,theta] = gradient(img)

    
    %Sobel kernels
    Kx = [-1 0 1; -2 0 2; -1 0 1];
    Ky = [-1 -2 -1; 0 0 0; 1 2 1];

    %convolution
    %Ix = conv2(img,Kx);
    %Iy = conv2(img,Ky);
    
    [rows, columns, ~] = size(img);
    
    %convolution x axis
    Ix = zeros(rows,columns);
    for i = 2:(columns-1)
        for j = 2:(rows-1)
            for u = -1:1
                for v = -1:1
                    Ix(i,j) = Ix(i,j) + Kx(u+2,v+2) * img(i-u,j-v);
                end
            end
        end
    end
    
    %convolution y axis
    Iy = zeros(rows,columns);
    for i = 2:(columns-1)
        for j = 2:(rows-1)
            for u = -1:1
                for v = -1:1
                    Iy(i,j) = Iy(i,j) + Ky(u+2,v+2) * img(i-u,j-v);
                end
            end
        end
    end
    
    %figure, imshow(Ix);
    %figure, imshow(Iy);

    G = hypot(Ix,Iy);
    G = G / max(max(G));
    theta = atan2(Iy, Ix);
    
    %figure, imshow(G);
    %figure, imshow(theta);

end