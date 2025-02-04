spm('defaults', 'eeg');

% Initialize the struct for SPM

S = [];
S.data = '/media/sarvagya-pc/2TB HDD/Balgrist/OPM/sub-001/ses-001/meg/sub-001_ses-001_task-mns_run-001_meg.bin';
S.channels = '/media/sarvagya-pc/2TB HDD/Balgrist/OPM/sub-001/ses-001/meg/sub-001_ses-001_task-mns_run-001_channels.tsv';
S.positions = '/media/sarvagya-pc/2TB HDD/Balgrist/OPM/sub-001/ses-001/meg/sub-001_ses-001_task-mns_run-001_positions.tsv';
S.meg = '/media/sarvagya-pc/2TB HDD/Balgrist/OPM/sub-001/ses-001/meg/sub-001_ses-001_task-mns_run-001_meg.json';
% S.path = '';
D = spm_opm_create(S);

% Downsample the data

S = [];
S.D = D;
S.method = 'resample';
S.fsample_new = 3000;
S.prefix = 'downsamples_data';
D = spm_eeg_downsample(S);

% Apply syntheric Gradiometry:

% S = [];
% S.D = D;
% S.gradient = [3];
% S.method = 'fieldtrip';
% S.prefix = 'g_';
% D = spm_eeg_ctf_synth_gradiometer(S);

S = [];
S.D = D;
S.confounds = {'*N0*', '*N4*'};
S.Y = 'MEGMAG';
D = spm_opm_synth_gradiometer(S);

% Apply bandpass filter between 20-300 Hz

S = [];
S.D = D;
S.freq = [20];
S.band = 'high';
D = spm_eeg_ffilter(S);

S = [];
S.D = D;
S.freq = [300];
S.band = 'low';
D = spm_eeg_ffilter(S);




% Linear regression is needed. How do I know which two reference labels?

% Notch filter for line noise or SG here. Discuss

S = [];
S.D = D;
S.freq = [47, 53];
S.band = 'stop';
D = spm_eeg_ffilter(S);

S = [];
S.D = D;
S.freq = [97 103];
S.band = 'stop';
D = spm_eeg_ffilter(S);

% For epochs, Do i need to use the trigger channel and check when it's
% applied? It says -100 to 300 ms stimulus onset. Can I use the one from
% the website?

S =[];
S.D=D;
S.timewin=[-100 300];
S.triggerChannels ={'NI-TRIG'};
S.thresh = 3;
S.bc = 1;
D= spm_opm_epoch_trigger(S);

S=[];
S.D = D;
S.timewin = [-100 0];
D = spm_eeg_bc(S);

% S = [];
% S.D =D;
% D = spm_eeg_average(S);

chans_to_plot = (1:95);

SE = std(D(1:95,:,:), [], 3)./sqrt(size(D,3));
mD = mean(D(1:95,:,:),3);
t_stat = mD(:,:,1)./SE;
plot(D.time, t_stat);

% SE = std(D(:,:,:), [], 3)./sqrt(size(D,3));
% t_stat = D(:,:,1)./SE;
% plot(mD.time, t_stat);

% D = spm_eeg_load('/home/neuroimaging/Desktop/Sarvagya_work/OPM-MEG/sub-001/ses-001/meg/sub-001_ses-001_task-mns_run-001_meg');

% PSD

S=[];
S.triallength = 3000; 
S.plot=1;
S.D=mD;
[~,freq]=spm_opm_psd(S);
ylim([1,1e5])
xlim([0,500])