% Copyright 2016 Google Inc.
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
% http ://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.

% Run bilateral guided upsampling pipeline on downsampled input and output, then
% compare to ground truth.
%
% input_ds: downsampled input
% edge_ds: downsampled edge
% output_ds: downsampled output
% weight_ds: downsampled weight map
% input_fs: full-size input
% edge_fs: full-size edge
%
% output_fs: ground-truth full-size output
% [optional] grid_size: affine bilateral grid size to pass through
%   [height width depth num_output_channels num_input_channels]
% [optional] lambda_s: spatial smoothness parameter to pass through
% [optional] intensity_options: intensity parameters to pass through
function result = testBGU_modified(input_ds, edge_ds, output_ds, weight_ds, ...
    input_fs, edge_fs, grid_size, lambda_s, intensity_options)


if ~exist('grid_size', 'var')
    grid_size = round([input_height / 16, input_width / 16, 8, ...
    output_channels, input_channels + 1]);
end

if ~exist('lambda_s', 'var')
    lambda_s = [];
end

if ~exist('intensity_options', 'var')
    intensity_options = [];
end
[gamma, A, b, lambda_s, intensity_options] = ...
    bguFit(input_ds, edge_ds, output_ds, weight_ds, grid_size, lambda_s, intensity_options);
result =  bguSlice(gamma, input_fs, edge_fs);

