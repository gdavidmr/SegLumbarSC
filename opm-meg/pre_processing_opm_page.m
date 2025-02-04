spm('defaults', 'eeg');

% Initialize the struct for SPM

S = [];
S.data = '/Users/sarvagyagupta/Desktop/college/PhD/Balgrist/ETH/PhD_work/github/sub-001/ses-001/meg/sub-001_ses-001_task-mns_run-001_meg.bin';
S.channels = '/Users/sarvagyagupta/Desktop/college/PhD/Balgrist/ETH/PhD_work/github/sub-001/ses-001/meg/sub-001_ses-001_task-mns_run-001_channels.tsv';
S.positions = '/Users/sarvagyagupta/Desktop/college/PhD/Balgrist/ETH/PhD_work/github/sub-001/ses-001/meg/sub-001_ses-001_task-mns_run-001_positions.tsv';
S.meg = '/Users/sarvagyagupta/Desktop/college/PhD/Balgrist/ETH/PhD_work/github/sub-001/ses-001/meg/sub-001_ses-001_task-mns_run-001_meg.json';
% S.path = '';
D = spm_opm_create(S);

% Downsample the data

S = [];
S.D = D;
S.method = 'resample';
S.fsample_new = 3000;
S.prefix = 'downsamples_data';
D = spm_eeg_downsample(S);


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

% Spatial models

S = [];
S.D = D;
% S.positions = '/home/neuroimaging/Desktop/Sarvagya_work/OPM-MEG/sub-001/ses-001/meg/sub-001_ses-001_task-mns_run-001_positions.tsv';
mD = spm_opm_amm(S);

% PSD

S=[];
S.triallength = 3000; 
S.plot=1;
S.D=D;
[~,freq]=spm_opm_psd(S);
ylim([1,1e5])
