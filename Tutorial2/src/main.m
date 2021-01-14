
%Get rgb and depth information folders
framesFolder = './assets/gait/gait_depth';
frames = dir(fullfile(framesFolder,'gait_depth_60frames_*.png'));
vr = VideoReader('./assets/gait/gait_RGB_60frames.avi');

for i = 1:numel(frames)
    % read rgb frame from video and respective depth image
    rgb_img = read(vr,i);    
    file_name_depth = fullfile(framesFolder,frames(i).name);
    depth_img = im2gray(imread(file_name_depth));
    
    img = feet_detection(rgb_img, depth_img);
    
    %Create name and right to file
    if i <= 10
        fileName = strcat('./output/frame0',int2str(i-1),'.png');
    else
        fileName = strcat('./output/frame',int2str(i-1),'.png');
    end  
    imwrite(img,fileName);
    
    [A,map] = rgb2ind(img,256);
    if i == 1
        imwrite(A,map,'.\output\og.gif','gif','LoopCount',Inf,'DelayTime',0.1);
    else
        imwrite(A,map,'.\output\og.gif','gif','WriteMode','append','DelayTime',0.1);
    end

end