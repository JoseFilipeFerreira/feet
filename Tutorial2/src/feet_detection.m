function [left_foot right_foot] = feet_detection(file_name_rgb, file_name_depth)

depth_img = im2gray(imread(file_name_depth));
rgb_img = im2gray(imread(file_name_rgb));


minx = 205;
maxx = 445;
miny = 80;
maxy = 380;

area_of_interest_depth = depth_img(miny:maxy, minx:maxx);

floor_per_line = specialMax(area_of_interest_depth, 100);

f = polyfit([0:size(floor_per_line)-1], floor_per_line,1);

compensated_floor_per_line = transpose(polyval(f, [0:size(floor_per_line) - 1]));

floor_per_point = repmat(compensated_floor_per_line, 1, maxx - minx + 1);

area_of_interest_depth(area_of_interest_depth > floor_per_point - 100) = 100000;
area_of_interest_depth(area_of_interest_depth < floor_per_point - 300) = 100000;

area_of_interest_bin = imbinarize(area_of_interest_depth);
figure, imshow(area_of_interest_bin)

area_of_interest_bin_closed = imclose(area_of_interest_bin,strel('square',15));
area_of_interest_bin_dilated = imdilate(area_of_interest_bin_closed,strel('square',9));
% figure, imshow(area_of_interest_bin)

[LToe,LHeel,RToe,RHeel] = findFeet(area_of_interest_bin_dilated);

x1 = [ LToe(1) , LHeel(1) ];
y1 = [ LToe(2) , LHeel(2) ];
x2 = [ RToe(1) , RHeel(1) ];
y2 = [ RToe(2) , RHeel(2) ];

figure,imshow(area_of_interest_depth)
hold on
plot(y1, x1, 'green')
plot(y2, x2, 'green')
hold off

% figure,imshow(depth_img)
% x = [minx maxx maxx minx minx];
% y = [maxy maxy miny miny maxy];
% imshow(rgb_img)
% hold on
% plot(x, y, 'red', 'LineWidth', 4)
end