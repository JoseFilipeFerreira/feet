
D = '.\assets\60_frames\gait_depth2'; %gait_depth
S = dir(fullfile(D,'gait_depth_60frames_*.png')); % pattern to match filenames.

for k = 1:numel(S)
    F = fullfile(D,S(k).name);
    img = feet_detection('', F);
    %imshow(img)
end