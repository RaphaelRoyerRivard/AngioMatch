# AngioMatch
Collection of scripts used to synchronize, track and match multi-view angiographic sequences


## How to format new angiographic data
1. Run the `ecg_extractor.m` script with MatLab (with admin rights) to extract the ECG signals off the DICOM files.
2. Run the `parse_ecg.py` script to identify the R-peaks in the ECG signals (check the parameters at the beginning of the main).
3. Add code in the `copy_r-peaks_files.py` script to create the patient folders + series folders and copy the R-peaks files in them.
4. Export the angiographic data from the DICOM files to jpeg images with the "Export to image" feature of the MicroDicom application.
5. Make sure the patients folder of the exported data have the same name as the patient folders where the R-peaks files were copied.
5. Add code in the `copy_angiographic_data.py` script to copy the sequence images to the final patient folders.
6. Run the `identify_relevant_frames.py` script that will parse the sequence images to create a file that specify the relevant frames of the sequences.
7. Run the `clean_angiographic_data.py` script to remove sequences where we couldn't identify the relevant frames.
