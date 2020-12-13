function [noisy_img] = noise(gray_scale_img, noise_type, noise_params)
%NOISE 
if noise_type == "gaussian"
    noise = rand(size(gray_scale_img)) * (noise_params * 2) - noise_params;
    noisy_img = gray_scale_img + noise;
elseif noise_type == "salt-n-pepper"
    noisy_img = gray_scale_img .* (rand(size(gray_scale_img)) >= (noise_params/2));
    noisy_img(rand(size(gray_scale_img)) <= (noise_params/2)) = 1;
else
    error("Unknow type of noise");
end

end

