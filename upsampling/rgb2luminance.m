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

% Convert rgb to luma by taking a dot product with coeffs.
%
% rgb should be height x width x 3.
% coeffs should be a 3-vector.
function luma = rgb2luminance(rgb, coeffs)

if ismatrix(rgb)
    luma = rgb;
    return;
end

if ~exist('coeffs', 'var')
    coeffs = [0.25, 0.5, 0.25];
end

if ndims(rgb) ~= 3
    error('rgb should be height x width x 3.');
end

if numel(coeffs) ~= 3
    error('coeffs must be a 3-element vector.');
end

if abs(1 - sum(coeffs)) > 1e-6
    warning('coeffs sum to %f, which is not 1.', sum(coeffs));
end

luma = coeffs(1) * rgb(:,:,1) + coeffs(2) * rgb(:,:,2) + ...
    coeffs(3) * rgb(:,:,3);
