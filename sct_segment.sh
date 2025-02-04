#!/bin/bash

subs=()
# search_dir=/media/sarvagya-pc/2TB\ HDD/Balgrist/GM_mask/01_data
# search_dir=/home/sarvagya-pc/Desktop/Balgrist_neuroimg/for_test
search_dir=/media/sarvagya-pc/2TB\ HDD/Balgrist/GM_mask/for_test
ses_folder=ses-
input_scan=/ses-01/anat/
file_scan="_desc-crop_T2starw.nii.gz"
# segment_output_deepseg_ca=seg_outputs/sct_deepseg_contrast_agnostic_output
# segment_output_deepseg=seg_outputs/sct_deepseg
# segment_output_unet=seg_outputs/sct_3d_Unet
# segment_output_deepseg_ca_20240930=seg_outputs/sct_deepseg_contrast_agnostic_20240930_output
totalsegout=seg_outputs/sct_deepseg_totalspineseg_output_0.75
out_folder="/home/sarvagya-pc/Desktop/Balgrist_neuroimg"

# This for loop stores all the subjects in an array 

for entry in "$search_dir"/*
do
  subs+=("$entry")
done


# subs=("${subs[@]:1}") #pops the first entry since it's not a subject folder

# subs=("${subs[@]:1}") #pops the first entry since it's not a subject folder

# echo "${subs[@]}"

for sub in "${subs[@]}"; do

	mapfile -t sessions < <(find "$sub" -maxdepth 1 -type d | grep "ses-")
	for ses in "${sessions[@]}"; do

		echo "The subjec is $sub"
		echo "The session is $ses"
		echo "The folder to process is $ses"/anat/
		# mkdir -p "$ses"/anat/"$segment_output_deepseg"
		# mkdir -p "$ses"/anat/"$segment_output_deepseg_ca"
		# mkdir -p "$ses"/anat/"$segment_output_unet"
		# mkdir -p "$ses"/anat/"$segment_output_deepseg_ca_20240930"
		echo mkdir -p "$out_folder/$totalsegout"
		mkdir -p "$out_folder/$totalsegout/${sub##*/}/${ses##*/}"

		search_directory="${ses}""/anat/" #Concats subject folder with the location of the nifti file in the subject
		echo "The search directory is $search_directory"
		echo $file_scan
		mapfile -t files_with_text < <(find "$search_directory" -type f | grep "$file_scan") #Find all files in the folder with subtext (Here it should be cropped T2S file)
		echo ${files_with_text[@]}
		for file in "${files_with_text[@]}"; do 

			echo "The file is $file"
			filename="${file##*/}" #get the filename from the whole text since this will add the whole location
			echo "The file name is: "${filename::-14}
			# sct_deepseg_sc \
			# -i "$ses"/anat/"${filename::-4}.nii" \
			# -c t2s \
			# -o "$ses"/anat/"$segment_output_deepseg"/"${filename::-4}_SctDeepSeg.nii" \

			# sct_deepseg \
			# -i "$ses"/anat/"${filename::-4}.nii" \
			# -o "$ses"/anat/"$segment_output_deepseg_ca"/"${filename::-4}_SctDeepSegContrastAgnostic.nii" \
			# -task seg_sc_contrast_agnostic

			# sct_deepseg \
			# -i "$ses"/anat/"${filename::-4}.nii" \
			# -o "$ses"/anat/"$segment_output_unet"/"${filename::-11}seg-Sct3DUnet_label-SC_mask.nii" \
			# -task seg_lumbar_sc_t2w

			# sct_deepseg \
			# -task seg_sc_contrast_agnostic \
			# -i "$ses"/anat/"${filename::-4}.nii" \
			# -o "$ses"/anat/"$segment_output_deepseg_ca_20240930"/"${filename::-11}seg-SctDeepSegCA20240930_label-SC_mask.nii"

			CUDA_VISIBLE_DEVICES=1 SCT_USE_GPU=1 sct_deepseg \
			-task totalspineseg \
			-i "$ses"/anat/"${filename::-7}.nii.gz" \
			-o "$out_folder/$totalsegout"/"${sub##*/}/${ses##*/}/${filename::-14}seg-SctDeepSegTotalSeg_label-SC_mask.nii"\
			-thr 0.75

			man_seg_filename="desc-crop_seg-manual_label-SC_mask.nii"
			man_seg=$(find "$search_directory" -type f | grep "$man_seg_filename")
			man_seg_file="${man_seg##*/}"

			# # sct_dice_coefficient \
			# # -i "$ses"/anat/"$segment_output_deepseg"/"${filename::-4}_SctDeepSeg.nii" \
			# # -d "$search_directory"/"$man_seg_file" \
			# # -2d-slices 2 \
			# # -o "$ses"/anat/"$segment_output_deepseg"/"DsOutput.txt"

			# # sct_dice_coefficient \
			# # -i "$ses"/anat/"$segment_output_deepseg_ca"/"${filename::-4}_SctDeepSegContrastAgnostic.nii" \
			# # -d "$search_directory"/"$man_seg_file" \
			# # -2d-slices 2 \
			# # -o "$ses"/anat/"$segment_output_deepseg_ca"/"DsOutput.txt"

			# # sct_dice_coefficient \
			# # -i "$ses"/anat/"$segment_output_unet"/"${filename::-11}seg-Sct3DUnet_label-SC_mask.nii" \
			# # -d "$search_directory"/"$man_seg_file" \
			# # -2d-slices 2 \
			# # -o "$ses"/anat/"$segment_output_unet"/"DsOutput.txt"

			# sct_dice_coefficient \
			# -i "$ses"/anat/"$segment_output_deepseg_ca_20240930"/"${filename::-11}seg-SctDeepSegCA20240930_label-SC_mask.nii" \
			# -d "$search_directory"/"$man_seg_file" \
			# -2d-slices 2 \
			# -o "$ses"/anat/"$segment_output_unet"/"DsOutput.txt"

			sct_dice_coefficient \
			-i "$out_folder/$totalsegout/${sub##*/}/${ses##*/}"/"${filename::-14}seg-SctDeepSegTotalSeg_label-SC_mask_step1_cord.nii" \
			-2d-slices 2 \
			-d "$search_directory"/"$man_seg_file" \
			-o "$out_folder/$totalsegout"/"${sub##*/}/${ses##*/}"/"DsOutput.txt"
			# -i "$out_folder/$totalsegout"/"${filename::-11}seg-SctDeepSegTotalSeg_label-SC_mask.nii" \

		done
	done
done