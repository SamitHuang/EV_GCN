from nilearn import datasets
import ABIDEParser as Reader
import os
import shutil

# Selected pipeline
pipeline = 'cpac'

# Input data variables
num_subjects = 871  # Number of subjects
#root_folder = '/bigdata/fMRI/ABIDE/'
root_folder = './'
data_folder = os.path.join(root_folder, 'ABIDE_pcp/cpac/filt_noglobal')

# Files to fetch
files = ['rois_ho']

filemapping = {'func_preproc': 'func_preproc.nii.gz',
               'rois_ho': 'rois_ho.1D'}

if not os.path.exists(data_folder): os.makedirs(data_folder)
shutil.copyfile('./subject_IDs.txt', os.path.join(data_folder, 'subject_IDs.txt'))

# Download database files
abide = datasets.fetch_abide_pcp(data_dir=root_folder, n_subjects=num_subjects, pipeline=pipeline,
                                 band_pass_filtering=True, global_signal_regression=False, derivatives=files)


subject_IDs = Reader.get_ids(num_subjects)
subject_IDs = subject_IDs.tolist()

# Create a folder for each subject
for s, fname in zip(subject_IDs, Reader.fetch_filenames(subject_IDs, files[0])):
    subject_folder = os.path.join(data_folder, s)
    if not os.path.exists(subject_folder):
        os.mkdir(subject_folder)

    # Get the base filename for each subject
    base = fname.split(files[0])[0]

    # Move each subject file to the subject folder
    for fl in files:
        if not os.path.exists(os.path.join(subject_folder, base + filemapping[fl])):
            shutil.move(base + filemapping[fl], subject_folder)

time_series = Reader.get_timeseries(subject_IDs, 'ho')

# Compute and save connectivity matrices
for i in range(len(subject_IDs)):
        Reader.subject_connectivity(time_series[i], subject_IDs[i], 'ho', 'correlation')

