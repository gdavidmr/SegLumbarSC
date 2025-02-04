% Create subject info for spinal OPM recordings 
% sub, sess and runs are strings for the subject or session e.g. '01'
% task is a string for the task, e.g. mns, squeeze, tns

function ss = createspinalsubs(sub,sess,run,task)

ss.sub = sub;
ss.sess = sess;
ss.rawDat = ['E:\Lydia\spinal_opm\rawData\sub-0' sub '\ses-0' sess '\meg\sub-0' sub '_ses-0' sess '_task-' task '_run-0' run '_meg.bin'];
ss.rawMeg = ['E:\Lydia\spinal_opm\rawData\sub-0' sub '\ses-0' sess '\meg\sub-0' sub '_ses-0' sess '_task-' task '_run-0' run '_meg.json'];
ss.rawChans = ['E:\Lydia\spinal_opm\rawData\sub-0' sub '\ses-0' sess '\meg\sub-0' sub '_ses-0' sess '_task-' task '_run-0' run '_channels.tsv'];
ss.rawPos = ['E:\Lydia\spinal_opm\rawData\sub-0' sub '\ses-0' sess '\meg\sub-0' sub '_ses-0' sess '_task-' task '_run-0' run '_positions.tsv'];
ss.outPath = ['E:\Lydia\spinal_opm\saveData\sub' sub '\ses' sess '\'];
ss.outFile = ['sub-0' sub '_ses-0' sess '_task-' task '_run-0' run '_spm'];
% ss.outPath = 'E:\Lydia\spinal_opm\saveData\DeleteThis\';

if strcmp(sub,'01')
    if strcmp(sess,'01')
        ss.spineChans = {'G2-DM','G2-18','G2-OH','G2-1B','G2-A4','G2-MW','G2-1A','G2-MX','G2-DH','G2-A8','G2-DQ',...
            'G2-DU','G2-DL','G2-DG','G2-A2','G2-MV','G2-N3','G2-OK','G2-DJ','G2-OI','G2-DI'};
        ss.headChans = {};
    elseif strcmp(sess,'03') | strcmp(sess,'04') | strcmp(sess,'05')
        ss.spineChans = {'G2-DM','G2-18','G2-OH','G2-1B','G2-A4','G2-MW','G2-1A','G2-MX','G2-DH','G2-A8','G2-DQ',...
            'G2-DU','G2-DL','G2-DG','G2-A2','G2-MV','G2-N3','G2-OK','G2-DJ','G2-OI','G2-DI',...
            'G2-DS','G2-DK','G2-DR','G2-OG','G2-A1','G2-DO'};
        ss.headChans = {'G2-OJ','G2-A0','G2-N2','G2-OF','G2-MY','G2-17','G2-A9','G2-1C','G2-MZ','G2-A3','G2-A7',...
            'G2-AA','G2-MT','G2-19','G2-A6'};
    elseif strcmp(sess,'06') % 1-4 right and 5-8 left 
        ss.spineChans = {'G2-MV','G2-1B','G2-A5','G2-DT','G2-DQ','G2-OI','G2-A1','G2-A2','G2-AA','G2-OK','G2-MW',...
            'G2-DR','G2-DO','G2-35','G2-OJ','G2-MY','G2-DS','G2-A4','G2-OH','G2-N2','G2-1C','G2-18'};
        ss.headChans = {'G2-A8','G2-A6','G2-17','G2-DM','G2-MZ','G2-N3','G2-MT','G2-A9','G2-DU','G2-DG','G2-MX','G2-DI','G2-A3',...
            'G2-OG','G2-A0','G2-DL','G2-DJ','G2-A7'};
    elseif strcmp(sess,'07') & (strcmp(task,'mns') | strcmp(task,'control'))
        ss.spineChans = {'G2-DU','G2-DL','G2-OG','G2-A8','G2-MX','G2-17','G2-MT','G2-DI','G2-DG','G2-A3','G2-MZ',...
            'G2-N3','G2-A0','G2-A9','G2-A6','G2-A7','G2-18','G2-A1','G2-MY','G2-DQ','G2-MV','G2-DT','G2-35','G2-OK'};%,...
%             'G2-1C','G2-OJ','G2-DO','G2-N2'}; % These channels are neck (mullet cast)
        ss.headChans = {'G2-OI','G2-1B','G2-A2','G2-A5','G2-A4','G2-MW','G2-DR','G2-AA','G2-OH',...
            'G2-DS','G2-1C','G2-OJ','G2-DO','G2-N2'}; % These channels are neck (mullet cast)
    elseif strcmp(sess,'07') & strcmp(task,'tns')
        ss.spineChans = {'G2-DU','G2-DL','G2-OG','G2-A8','G2-MX','G2-17','G2-MT','G2-DI','G2-DG','G2-A3','G2-MZ',...
            'G2-N3','G2-A0','G2-A9','G2-A6','G2-A7','G2-18','G2-A1','G2-MY','G2-DQ','G2-MV','G2-DT','G2-35'}; %,'G2-OK'};
        ss.headChans = {'G2-1C','G2-OJ','G2-DO','G2-N2','G2-OI','G2-1B','G2-A2','G2-A5','G2-A4','G2-MW','G2-DR','G2-AA','G2-OH',...
            'G2-DS'};
    end
elseif strcmp(sub,'02') | strcmp(sub,'03') 
    % sub 02 - 1-4 right and 5-8 left
    % sub 03 - 2-5 left and 6-9 right 
    ss.spineChans = {'G2-N3','G2-MV','G2-MY','G2-DG','G2-A1','G2-OG','G2-DI','G2-MX','G2-MZ','G2-A6','G2-17',...
        'G2-MT','G2-A3','G2-A8'};
    ss.headChans = {};  
elseif strcmp(sub,'04') % 1-4 are left, 5-8 right MNS
    ss.spineChans = {'G2-MV','G2-1B','G2-A5','G2-DT','G2-DQ','G2-OI','G2-A1','G2-A2','G2-AA','G2-OK','G2-MW',...
        'G2-DR','G2-DO','G2-35','G2-OJ','G2-MY','G2-DS','G2-A4','G2-OH','G2-N2','G2-1C','G2-18'};
    ss.headChans = {'G2-A8','G2-A6','G2-17','G2-DM','G2-MZ','G2-N3','G2-MT','G2-A9','G2-DU','G2-DG','G2-MX','G2-DI','G2-A3',...
        'G2-OG','G2-A0','G2-DL','G2-DJ','G2-A7'};
elseif strcmp(sub,'05') % 1-4 are left, 5-8 right MNS
    ss.spineChans = {'G2-MV','G2-1B','G2-A5','G2-DT','G2-DQ','G2-OI','G2-A1','G2-A2','G2-AA','G2-OK','G2-MW',...
        'G2-DR','G2-DO','G2-35','G2-OJ','G2-MY','G2-DS','G2-A4','G2-OH','G2-N2','G2-1C','G2-18'};
    ss.headChans = {}; 
end
    

end