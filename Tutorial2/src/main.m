
D = '.\assets\60_frames\gait_depth2'; %gait_depth
S = dir(fullfile(D,'gait_depth_60frames_*.png')); % pattern to match filenames.
vr = VideoReader('.\assets\60_frames\gait_RGB_60frames.avi');

for k = 1:numel(S)
    rgb_img = read(vr,k);    
    file_name_depth = fullfile(D,S(k).name);
    depth_img = im2gray(imread(file_name_depth));
    
    img = feet_detection(rgb_img, depth_img);
    %imshow(img)
end