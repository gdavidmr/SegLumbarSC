file_1 = '/media/sarvagya-pc/2TB HDD/Balgrist/THS/New_Pipeline/BCN/S01/M01/Results/wp1s2017-10-14_07-54-085400-00001-00176-1_RFSC_MTsat.nii';
file_2 = "/media/sarvagya-pc/2TB HDD/Balgrist/THS/New_Pipeline/BCN/S02/M01/Results/wp1s2017-10-14_09-03-093649-00001-00176-1_RFSC_MTsat.nii";
% file_1_MPM = "/Users/sarvagyagupta/Desktop/college/PhD/Balgrist/ETH/PhD_work/MPM/BSL-002/ses-01/Results/wp1_sub_BSL-002_ses-01_map_MTsat_masked_thresh.nii";
% file_2_MPM = "/Users/sarvagyagupta/Desktop/college/PhD/Balgrist/ETH/PhD_work/MPM/BSL-002/ses-02/Results/wp1_sub_BSL-002_ses-02_map_MTsat_masked_thresh.nii";
file_nifti_1 = niftiread(file_1);
file_nifti_2 = niftiread(file_2);
% file_nifti_1 = uint8(niftiread(file_1_MPM));
% file_nifti_2 = uint8(niftiread(file_2_MPM));
slices = size(file_nifti_1,3);
sub_1_ses_1 = [];
sub_1_ses_2 = [];

for i=1:slices

    img_1 = file_nifti_1(:,:,i);
    img_2 = file_nifti_2(:,:,i);
    % img_1 = (file_nifti_1(:,:,i));
    % img_2 = (file_nifti_2(:,:,i));
    
    % rgbImage = ind2rgb(img, gray);
    % rgbImage_1 = imrotate(img_1, 90);
    % rgbImage_2 = imrotate(img_2, 90);
    % formatSpec = '%03d\n';
    % filename = strcat("/Users/sarvagyagupta/Desktop/college/PhD/Balgrist/ETH/PhD_work/THS/Images/M02/axial/",num2str(i,formatSpec),".png");
    % imwrite(rgbImage, filename)
    img_flatten_1 = reshape(img_1.',1,[]);
    img_flatten_2 = reshape(img_2.', 1, []);
    sub_1_ses_1 = [sub_1_ses_1 img_flatten_1];
    sub_1_ses_2 = [sub_1_ses_2 img_flatten_2];

end

sub_1_ses_1_pca = double(sub_1_ses_1);
sub_1_ses_2_pca = double(sub_1_ses_2);

corrcoef(sub_1_ses_1, sub_1_ses_2)