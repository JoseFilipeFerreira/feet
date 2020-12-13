function [] = main_smoothfilters(filename, noise_type, noise_params, filtering_domain, smoothing_type, a, z)
%main_smoothfilters
[filepath,name,~] = fileparts(filename);

gray_scale_img = imread(filename);

noisy_img = noise(im2double(gray_scale_img), noise_type, noise_params);

smooth_img = noisy_img;
if filtering_domain == "spatial"
    smooth_img = im2double(smooth_img);
    matrix = ones(a);

    if smoothing_type == "average"
        for i = 1:a
            for j = 1:a
                matrix(i,j) = 1/(a*a);
            end
        end

        smooth_img = conv2(smooth_img,matrix);

        [p3, p4] = size(smooth_img);
        [q1, q2] = size(gray_scale_img);
        q1 = q1 - 1;
        q2 = q2 - 1;
        i3_start = floor((p3-q1)/2);
        i3_stop = i3_start + q1;
        i4_start = floor((p4-q2)/2);
        i4_stop = i4_start + q2;

        smooth_img = smooth_img(i3_start:i3_stop, i4_start:i4_stop, :);

    elseif smoothing_type == "gaussian"
        h = fspecial('gaussian',z,a);
        smooth_img = imfilter(smooth_img, h);

    elseif smoothing_type == "median"
        [p,q] = size(matrix);
        smooth_img = medfilt2(smooth_img,[p q]);
    else
        error("Unknow type of smothing");
    end

elseif filtering_domain == "frequency"
    f = im2double(smooth_img);
        [M, N] = size(f);
        P = 2*M;
        Q = 2*N;
        f = padarray(f,[M N],'post');

        for i = 1:P
            for j = 1:Q
                f(i,j) = (-1)^(i+j) * f(i,j);
            end
        end

        G = fft2(f);

        if smoothing_type == "gaussian"
            H = fspecial('gaussian',[P Q],a);
            H = mat2gray(H);
        elseif smoothing_type == "Butterworth"
            H = double(zeros(P,Q));
            filter_order = a;
            cutoff = z;

            for i=1:P
                for j=1:Q
                    dist = ((i-P/2)^2 + (j-Q/2)^2)^0.5;
                    H(i,j) = 1 / (1 + (dist/cutoff)^(2*filter_order));
                end
            end
        else
            error("Unknow type of smothing");
        end

        G = H .* G;

        smoothed_image = (ifft2(G));
        smoothed_image = real(smoothed_image);

        for i=1:M
            for j=1:N
                smoothed_image(i,j) = (-1)^(i+j) * smoothed_image(i,j);
            end
        end

        smooth_img = smoothed_image(1:M, 1:N);

imwrite(noisy_img, filepath+"/"+name+"_"+noise_type+"_"+noise_params+".png", "png");

imwrite(smooth_img, filepath + "/" + name + "_smooth_" + filtering_domain + "_" + smoothing_type + "_" + a + "_" + z + ".png", "png");

end
