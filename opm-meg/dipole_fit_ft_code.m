% D should be preprocessed data from spm, i.e. after filtering and AMM

%% Pick a peak 
% You could also do this manually, you just need fittime to be a 1x2 vector 
% for the start and end of the time you want to fit the dipole to in seconds

[maxvals,maxtimes] = max(abs(D(indchantype(D, 'MEGMAG', 'GOOD'),:,1)),[],2);
[~, fitind] = max(maxvals);
fitind = maxtimes(fitind);
fittime=[D.time(fitind) D.time(fitind+1)];

%% Dipole fit

data = fttimelock(D);

sensor_data = D.sensors('MEG');
x_coord = sensor_data.chanpos(:,1);
z_coord = sensor_data.chanpos(:,3);

% Prepare source model - grid needs to cover the area that could contain
% the dipole, in the same coordinate system as the sensor positions in
% data.grad (or equivalently D.sensors('MEG').chanpos)
cfg = [];
cfg.method = 'basedongrid';
% cfg.xgrid = -100:5:100;
cfg.xgrid = mean(x_coord)-25:5:mean(x_coord)+25;
% cfg.ygrid = -100:5:150;
cfg.ygrid = -650:10:-525;
% cfg.zgrid = -100:5:100;
cfg.zgrid = min(z_coord):10:max(z_coord);
cfg.unit = 'mm';
src = ft_prepare_sourcemodel(cfg);

% Prepare head model
cfg = [];
cfg.method = 'infinite';
cfg.grad = data.grad;
hdm = ft_prepare_headmodel(cfg,src);

% Dipole fit
cfg = [];
cfg.headmodel = hdm;
cfg.sourcemodel = src;
cfg.grad = data.grad;
cfg.latency = fittime;
cfg.gridsearch = 'yes';
results = ft_dipolefitting(cfg,data);

%% Plot 

figure; 
ft_plot_sens(D.sensors('MEG'));
ft_plot_dipole(results.dip.pos, mean(results.dip.mom(1:3,:),2), 'unit', 'mm'); %
hold on;
% Put a rough scalp in to make easier to look at
scalp = ft_read_headshape(D.inv{1}.mesh.tess_scalp);
ft_plot_mesh(scalp, 'edgecolor', [0.7 0.7 0.7], 'facealpha', 0.3, 'edgealpha', 0.3);

final_lf = zeros(42, size(src.pos, 1)*3);
for i=1:size(src.pos, 1)
    final_lf(:, (3*i - 2):(3*i)) = ft_compute_leadfield(src.pos(i,:), D.sensors('MEG'), hdm);
end

p = eye(42) - (sensor_data.chanori*pinv(sensor_data.chanori));
final_out = p*final_lf;
% corrcoef(final_lf, final_out)
corr_mat = [];
for i=1:size(final_out,2)
    val = corrcoef(final_out(:,i), final_lf(:,i));
    corr_mat = [corr_mat val(2,1)];
end

dip_inds_1 = 1:3:size(corr_mat,2); 
dip_inds_2 = 2:3:size(corr_mat,2);
dip_inds_3 = 3:3:size(corr_mat,2);

figure;
ft_plot_sens(D.sensors('MEG'));
hold on;
scatter3(src.pos(:, 1), src.pos(:, 2), src.pos(:, 3), [],corr_mat(dip_inds_1), 'filled')

figure;
ft_plot_sens(D.sensors('MEG'));
hold on;
scatter3(src.pos(:, 1), src.pos(:, 2), src.pos(:, 3), [],corr_mat(dip_inds_2), 'filled')

figure;
ft_plot_sens(D.sensors('MEG'));
hold on;
scatter3(src.pos(:, 1), src.pos(:, 2), src.pos(:, 3), [],corr_mat(dip_inds_3), 'filled')