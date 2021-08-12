function BGU(input_file_fs, output_fs, output_file_ds)
input_file_fs = char(input_file_fs);
output_fs = char(output_fs);
output_file_ds = char(output_file_ds);
input_fs = im2double(imread(input_file_fs));
edge_fs = rgb2luminance(input_fs); % Used to slice at full resolution.
output_ds = im2double(imread(output_file_ds));
if size(output_ds, 1) > 300 || size(output_ds, 2) > 300
    output_ds = imresize(output_ds,[300,300]);
end
input_ds = imresize(input_fs, [size(output_ds, 1), size(output_ds, 2)]);
edge_ds = rgb2luminance(input_ds); % Determines grid z at low resolution.
result = testBGU_modified(input_ds, edge_ds, output_ds, [], input_fs, edge_fs, []);

imwrite(result, output_fs);