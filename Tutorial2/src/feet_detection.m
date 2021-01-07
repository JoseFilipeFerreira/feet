function [] = feet_detection(file_name_rgb, file_name_depth)

depth_img = im2gray(imread(file_name_depth));
rgb_img = im2gray(imread(file_name_rgb));

[sx sy] = size(rgb_img);

figure,imshow(depth_img);
minx = 205
maxx = 445
miny = 60
maxy = 380
x = [minx maxx maxx minx minx];
y = [maxy maxy miny miny maxy];

imshow(rgb_img)
hold on
plot(x, y, 'red', 'LineWidth', 4)

floor_per_line = max(depth_img);
floor_per_line(ones(1,sy), :)

end

