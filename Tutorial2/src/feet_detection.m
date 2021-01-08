function [img] = feet_detection(rgb_img, depth_img)

minx = 205;
maxx = 445;
miny = 80;
maxy = 380;

area_of_interest_depth = depth_img(miny:maxy, minx:maxx);

floor_per_line = specialMax(area_of_interest_depth, 100);

f = polyfit([0:size(floor_per_line)-1], floor_per_line,1);

compensated_floor_per_line = transpose(polyval(f, [0 : size(floor_per_line) - 1]));

floor_per_point = repmat(compensated_floor_per_line, 1, maxx - minx + 1);

area_of_interest_depth(area_of_interest_depth > floor_per_point - 100) = 100000;

area_of_interest_depth(area_of_interest_depth < floor_per_point - 300) = 100000;

area_of_interest_bin = imbinarize(area_of_interest_depth);

area_of_interest_bin_closed = imclose(area_of_interest_bin,strel('square',15));
area_of_interest_bin_dilated = imdilate(area_of_interest_bin_closed,strel('square',9));

[LToe,LHeel,RToe,RHeel] = findFeet(area_of_interest_bin_dilated);

img = rgb_img;
img = insertShape(img, 'Rectangle', [ minx, miny, maxx-minx, maxy-miny], 'Color', 'red', 'LineWidth', 2);
img = insertShape(img,'Line',[LToe(2) + minx, LToe(1) + miny, LHeel(2) + minx, LHeel(1) + miny ], 'Color', 'green');
img = insertShape(img,'Line',[RToe(2) + minx, RToe(1) + miny, RHeel(2) + minx, RHeel(1) + miny ], 'Color', 'yellow');

end