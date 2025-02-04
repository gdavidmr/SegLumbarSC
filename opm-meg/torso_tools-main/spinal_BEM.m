%% script for spinal cord BEM forward modelling

addpath(genpath('/home/sarvagya-pc/Desktop/Balgrist_neuroimg/opm-meg/hbf_lc_p-master'))
% clearvars
% close all
% clc

% cd('D:\');
% Metadata;
% proj_init;
% cd('D:\Simulations\');
%% Load in all generated meshes/ inputs

% geoms_path = ('D:\Simulations\Example' );
% geoms = load(fullfile(geoms_path, 'BEM_inputs_Steph.mat'));
geoms = load(fullfile('BEM_inputs_Steph.mat'));
ordering = {'wm', 'bone', 'heart', 'lungs', 'torso'}; %will need to remove the bone if not including that for BEM

clear mesh
mesh_idx = 1;

% tt_add_bem;

for ii = 1:numel(ordering)

    field = ['mesh_' ordering{ii}];

    tmp = [];
    tmp.faces         = geoms.(field).faces;
    tmp.vertices         = geoms.(field).vertices;
    tmp.unit        = 'm'; 
    tmp.name        = ordering{ii};

    orient = hbf_CheckTriangleOrientation(tmp.vertices,tmp.faces);
        if orient == 2
            tmp.faces = tmp.faces(:,[1 3 2]);
        end

    mesh(ii) = tmp;

end

cord_mesh = mesh(1);
bone_mesh = mesh(2); %remove bone here
heart_mesh = mesh(3);
lung_mesh = mesh(4);
torso_mesh = mesh(5);

%% Generate source and grad structures
src = [];
src.pos = geoms.sources.pos;
src.inside = ones(length(src.pos),1);
src.unit = 'm'; 

% Generate the grad structure for the back
grad_back = [];
sensor_data = D.sensors('MEG');
theta = pi; % 180 degrees in radians
rotation_matrix = [cos(theta), -sin(theta), 0;
                   sin(theta),  cos(theta), 0;
                   0,           0,          1]; % Rotation about Z-axis

% Apply the rotation
rotated_data = (rotation_matrix * sensor_data.chanpos')'; % Transpose for matrix multiplication
rotated_ori = (rotation_matrix * sensor_data.chanori')';
translation_matrix = [1, 0, 0, 0.8;
                      0, 1, 0, -0.45;
                      0, 0, 1, -0.2;
                      0, 0, 0, 1];
tform = [-850 600 275];

transformedmesh = minus(rotated_data, tform);
% grad_back.coilpos = geoms.back_coils_3axis.positions;
% grad_back.coilori = geoms.back_coils_3axis.orientations;
grad_back.coilpos = transformedmesh/1000;
grad_back.coilori = rotated_ori;
grad_back.tra = eye(length(grad_back.coilpos));
grad_back.label = sensor_data.label;
% for ii = 1:length(grad_back.coilpos)
%     grad_back.label{ii} = sprintf('Chan-%03d',ii);
% end
grad_back.unit = 'm';

% % Generate the grad structure for the front
% grad_front = [];
% grad_front.coilpos = geoms.front_coils_3axis.positions;
% grad_front.coilori = geoms.front_coils_3axis.orientations;
% grad_front.tra = eye(length(grad_front.coilpos));
% for ii = 1:length(grad_front.coilpos)
%     grad_front.label{ii} = sprintf('Chan-%03d',ii);
% end
% grad_front.unit = 'm';

%% lets do some BEM modelling!

%first pick one point in source structure and check if its inside the bone
%or between segments
i = 23; %ranges between 1 and size of the src_grid
    
single_source_point = src.pos(i, :);

figure;
hold on;
% scatter3(single_source_point(:, 1), single_source_point(:, 2), single_source_point(:, 3), 'r', 'filled');
% patch('Faces', torso_mesh.faces, 'Vertices', torso_mesh.vertices, 'FaceColor', [0.9, 0.9, 0.9], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
% patch('Faces', cord_mesh.faces, 'Vertices', cord_mesh.vertices,'FaceColor', 'yellow', 'EdgeColor', 'none', 'FaceAlpha', 0.5);
% patch('Faces', bone_mesh.faces, 'Vertices', bone_mesh.vertices, 'FaceColor', 'cyan', 'EdgeColor', 'none', 'FaceAlpha', 0.2);
% axis off;
axis equal;
view(3); 
hold on;
% sensor_data_flipped = (sensor_data.chanpos);
% scatter3(rotated_data(:,1)/1000, rotated_data(:,2)/1000, rotated_data(:,3)/1000)
% hold on;
scatter3(transformedmesh(:,1)/1000, transformedmesh(:,2)/1000, transformedmesh(:,3)/1000)
hold on;
scatter3(src.pos(:,1), src.pos(:,2), src.pos(:,3))
xlabel('x'); ylabel('y'); zlabel('z')
ft_plot_sens(grad_back);
hold off;

cratio = 40;

Ls_back_temp = [];
Ls_back = [];
source_ori = [1,0,0;0,1,0;0,0,1];
for i=1:size(src.pos,1)
    i
    for j=1:3

% Forward model using BEM
S_forward = [];
% S_forward.pos = single_source_point;
S_forward.pos = src.pos(i, :);
S_forward.T = geoms.transform_matrix;                      
S_forward.ori = source_ori(j,:)  %1 A/mm --> [1,0,0] = left to right, [0,1,0] = anerior to posterior, [0,0,1] = superior to inferior
S_forward.posunits = 'm';         
S_forward.names = {'blood','lungs','torso'};  
S_forward.ci = [0.33 .33/cratio .62 .05 .23]; %conductivities with cord and bone
% S_forward.ci = [.62 .05 .23];
S_forward.co = [.23 .23 .23 .23 0 ]; %conductivities with cord and bone
% S_forward.co = [.23 .23 0];
S_forward.cord = cord_mesh;
S_forward.vertebrae = bone_mesh; %hash out of not including the bone


S_back = S_forward;
S_back.sensors = grad_back;
% Ls_back = tt_fwd_bem3(S_back);


Ls_back_temp = [Ls_back_temp tt_fwds_bem5(S_back)]; %with bone
    end
    Ls_back = [Ls_back Ls_back_temp];
    Ls_back_temp = [];
end
% Ls_back = reshape(Ls_back, [42,45]);
% S_front = S_forward;
% S_front.sensors = grad_front;
% % Ls_front = tt_fwd_bem3(S_front);
% Ls_front = tt_fwds_bem5(S_front); %with bone

%% split Ls_ structure into its relevent parts
% Ls_back_xyz = reshape(Ls_back, [size(grad_back.label, 1)/2 , 2]);
Ls_back_xyz = reshape(Ls_back, [size(grad_back.label, 1)/2 , size(Ls_back, 2), 2]);
% Ls_front_xyz = reshape(Ls_front, [size(grad_back.label, 1)/3 , 3]);

% Ls_back_x = Ls_back_xyz(:, 1);
% Ls_back_y = Ls_back_xyz(:, 2);
% Ls_back_z = Ls_back_xyz(:, 3);
src_top_idx = 1;
src_mid_idx = round(size(src.pos,1)/2);
src_bottom_idx = size(src.pos,1);

Ls_back_x_top = Ls_back_xyz(:,src_top_idx,1);
Ls_back_y_top = Ls_back_xyz(:,src_top_idx, 2);

Ls_back_x_mid = Ls_back_xyz(:,src_mid_idx,1);
Ls_back_y_mid = Ls_back_xyz(:,src_mid_idx, 2);

Ls_back_x_bottom = Ls_back_xyz(:,src_bottom_idx,1);
Ls_back_y_bottom = Ls_back_xyz(:,src_bottom_idx, 2);

Ls_back_rad = Ls_back(endsWith(grad_back.label, '-RAD'),:);

% Ls_front_x = Ls_front_xyz(:, 1);
% Ls_front_y = Ls_front_xyz(:, 2);
% Ls_front_z = Ls_front_xyz(:, 3);

%% now lets plot
cmap = jet;  % Colormap

back_pos = grad_back.coilpos(1:size(grad_back.label, 1) / 2, :);

% front_pos = grad_front.coilpos(1:size(grad_back.label, 1) / 3, :);

orientations = {'X', 'Y', 'Z'};  

figure;
tiledlayout(2, 3, 'Padding', 'compact', 'TileSpacing', 'compact');

% --- Plotting Back View ---
% for axis = 1:2
%     ax = nexttile;
% 
%     switch axis
%         case 1, leadfield_data = Ls_back_x;
%         case 2, leadfield_data = Ls_back_y;
%         case 3, leadfield_data = Ls_back_z;
%     end
% 
%     plot_topoplot_xz(ax, back_pos, leadfield_data, cmap); %you may need to check this function - has two versions and cant remeber which one is the correct one here
% 
%     title(sprintf('Back - %s Axis', orientations{axis}));
% end

for axis = 1:2
    ax = nexttile;

    switch axis
        case 1, leadfield_data = Ls_back_x_top;
        case 2, leadfield_data = Ls_back_y_top;
        % case 3, leadfield_data = Ls_back_z;
    end

    plot_topoplot_xz(ax, back_pos, leadfield_data, cmap); %you may need to check this function - has two versions and cant remeber which one is the correct one here

    title(sprintf('Back - %s Axis', orientations{axis}));
end

hold on;
scatter(transformedmesh(:,1)/1000, transformedmesh(:,3)/1000)

figure;
tiledlayout(2, 3, 'Padding', 'compact', 'TileSpacing', 'compact');

for axis = 1:2
    ax = nexttile;

    switch axis
        case 1, leadfield_data = Ls_back_x_mid;
        case 2, leadfield_data = Ls_back_y_mid;
        % case 3, leadfield_data = Ls_back_z;
    end

    plot_topoplot_xz(ax, back_pos, leadfield_data, cmap); %you may need to check this function - has two versions and cant remeber which one is the correct one here

    title(sprintf('Back - %s Axis', orientations{axis}));
end

figure;
tiledlayout(2, 3, 'Padding', 'compact', 'TileSpacing', 'compact');

for axis = 1:2
    ax = nexttile;

    switch axis
        case 1, leadfield_data = Ls_back_x_bottom;
        case 2, leadfield_data = Ls_back_y_bottom;
        % case 3, leadfield_data = Ls_back_z;
    end

    plot_topoplot_xz(ax, back_pos, leadfield_data, cmap); %you may need to check this function - has two versions and cant remeber which one is the correct one here

    title(sprintf('Back - %s Axis', orientations{axis}));
end

% --- Plotting Front View ---
% for axis = 1:3
%     ax = nexttile;
% 
%     switch axis
%         case 1, leadfield_data = Ls_front_x;
%         case 2, leadfield_data = Ls_front_y;
%         case 3, leadfield_data = Ls_front_z;
%     end

    % plot_topoplot_xz(ax, front_pos, leadfield_data, cmap);
    % 
    % title(sprintf('Front - %s Axis', orientations{axis}));
% end

sgtitle('Leadfield Distribution of Dipole Pointed Anterior to Posterior');
