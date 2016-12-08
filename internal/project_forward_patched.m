function [sensor_image] = project_forward_patched(pixels, volume_n_px, n_px_per_ml, H)

%% find microlens-multiples that contain all the pixels
px_min = min(pixels, [], 1);
px_max = max(pixels, [], 1);
% number of microlenses left of the first one in the patch. lowest value can be zero, largest is total number of MLs minus 1
offset_mls_lateral = floor((px_min(1:2)-1) / n_px_per_ml);
% number of pixels to the left of the first one in the patch. lowest value can be zero, largest is (total number of MLs minus 1) * n_px_per_ml
offset_px_lateral = offset_mls_lateral * n_px_per_ml;
n_ml_lateral = ceil(px_max(1:2) / n_px_per_ml) - offset_mls_lateral;
% always make n_ml_lateral odd, to have a well-defined center pixel
if mod(n_ml_lateral(1), 2) == 0
    n_ml_lateral(1) = n_ml_lateral(1) + 1;
end
if mod(n_ml_lateral(2), 2) == 0
    n_ml_lateral(2) = n_ml_lateral(2) + 1;
end
%disp(num2str(n_ml_lateral));
offset_z = px_min(3) - 1; % number of planes above the first one in the patch
n_z = px_max(3) - offset_z;

%% paint pixels into volume patch
volume = zeros([n_ml_lateral * n_px_per_ml, n_z]);
for px_ix = 1:size(pixels,1)
    volume(pixels(px_ix, 1) - offset_px_lateral(1), pixels(px_ix, 2) - offset_px_lateral(2), pixels(px_ix, 3) - offset_z) = 1;
end
%figure; imagesc(squeeze(volume(:,:,1))); axis image; colorbar

%% Do the forward projection of the patch
zerospace = zeros(size(volume, 1), size(volume, 2));
tempspace = zerospace;
totalprojection = [];
for i=1:n_px_per_ml
    for j=1:n_px_per_ml
        for k=1:size(volume, 3)
            tempspace = zerospace;
            tempspace(i:n_px_per_ml:end, j:n_px_per_ml:end) = volume(i:n_px_per_ml:end, j:n_px_per_ml:end, k);
            if isempty(totalprojection)
                totalprojection = conv2(tempspace, H(:,:,i,j,offset_z + k), 'full');
            else
                totalprojection = totalprojection + conv2(tempspace, H(:,:,i,j, offset_z + k), 'full');
            end
        end
    end
end

%% copy totalprojection into final sensor_image, centered around center of volume
% since n_ml_lateral is always odd, the size of totalprojection will always be odd (since kernelwidth // w * 2 is added
% to the volume width by conv2('full'))
pj_x = size(totalprojection, 1);
pj_y = size(totalprojection, 2);
center_x = offset_px_lateral(1) + ceil((n_ml_lateral(1) / 2) * n_px_per_ml);
center_y = offset_px_lateral(2) + ceil((n_ml_lateral(2) / 2) * n_px_per_ml);

% rectangle in sensor coordinates where we need to paint with the projection patch
% make sure to clip to size of image
x0 = max([1, center_x - floor(pj_x/2)]);
x1 = min([volume_n_px(1), center_x + floor(pj_x/2)]);
y0 = max([1, center_y - floor(pj_y/2)]);
y1 = min([volume_n_px(2), center_y + floor(pj_y/2)]);

% transform clipped rectangle in sensor coordinates to rectangle in patch coordinates
p0 = x0 - (center_x - floor(pj_x/2) - 1);
p1 = x1 - (center_x - floor(pj_x/2) - 1);
q0 = y0 - (center_y - floor(pj_y/2) - 1);
q1 = y1 - (center_y - floor(pj_y/2) - 1);

sensor_image = zeros(volume_n_px(1:2));
sensor_image(x0:x1, y0:y1) = totalprojection(p0:p1, q0:q1);
end


