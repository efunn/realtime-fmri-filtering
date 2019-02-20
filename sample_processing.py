import os # basic file and path operations
import numpy as np # math and arrays
from nilearn.image import load_img # basic image loading
from nilearn.input_data import NiftiMasker # image masking and more!

# set up basic directories
data_dir = 'Documents\\fmri' # directory where subj_id folder exists
subj_id = 'ff001' # subject ID in directory
sess_id = 'sess2' # session ID in BOLD subdirectory
bold_dir = os.path.join(data_dir,subj_id,'bold',sess_id)
ref_dir = os.path.join(data_dir,subj_id,'ref')

# select and load mask
mask_id = 'mask_lh_s1'
mask_file = os.path.join(ref_dir,mask_id+'.nii')
mask_img = load_img(mask_file)

# load raw data
raw_fmri_data = load_img(os.path.join(bold_dir,'*.nii'))

# mask the data (and nothing else)
# all of these masking options can take a minute or so
masker = NiftiMasker(mask_img=mask_img)
masker.fit(raw_fmri_data)
masked_fmri_data = masker.transform(raw_fmri_data)

# here's a fancier example of masking the data with built-in zscore and detrend
# we first need to create the 'run' labels
num_runs = 8; baseline_trs = 20; trials_per_run = 20; trs_per_trial = 8;
trs_per_run = baseline_trs + trials_per_run*trs_per_trial 
run_labels = np.repeat(range(num_runs),trs_per_run)
fancy_masker = NiftiMasker(mask_img=mask_img,
                           sessions=run_labels, # this is 'run' labels for zscore/detrend
                           standardize=True, # normalize/zscore
                           detrend=True) # linear detrend
fancy_masker.fit(raw_fmri_data)
fancy_masked_fmri_data = fancy_masker.transform(raw_fmri_data)


#####################################
# sample plotting of voxel activity #
#####################################
import matplotlib.pyplot as plt # for plotting

sample_voxel = 500 # random voxel index - once we get to decoding, pick an important one!
plt.ion() # this makes your plots appear immediately, otherwise need to call plt.show() after plt.plot()
basic_voxel_activity = masked_fmri_data[:,sample_voxel] # rows are time, columns are voxels
fancy_voxel_activity = fancy_masked_fmri_data[:,sample_voxel] # (note: python indexes from zero) 

from scipy.stats import zscore
plt.plot(fancy_voxel_activity)
plt.plot(zscore(basic_voxel_activity)) # just zscore for rough plotting
plt.xlabel('TR')
plt.ylabel('voxel activity')
plt.legend(['zscored by run','zscored over entire experiment'])


###################
# sample decoding #
###################
import pandas as pd # great tool for playing with spreadsheet-type data
# num_runs = 8; baseline_trs = 20; trials_per_run = 20; trs_per_trial = 8; # reminder!
# trs_per_run = baseline_trs + trials_per_run*trs_per_trial # also reminder!

# load in behavioral data
raw_pressing_data = pd.read_csv(os.path.join(ref_dir,'ft-data-'+sess_id+'.txt'))
raw_pressing_labels = raw_pressing_data.probe # 'probe' was the instructed finger

# generate trial labels (.reset_index() is a useful Pandas function to note)
presses_per_trial = 10
trial_labels = raw_pressing_labels[::presses_per_trial].reset_index().probe 

# choose TRs to decode from (this is kinda complicated, sorry!)
tr_to_extract = 3 # pull out the 3rd TR from each trial (or is it the 4th...?)
trial_trs_in_run = baseline_trs + np.arange(
    tr_to_extract,tr_to_extract+trs_per_trial*trials_per_run,trs_per_trial)
trs_to_label = (np.tile(trial_trs_in_run,num_runs)
    + np.repeat(trs_per_run*np.arange(num_runs),trials_per_run))

# extract only relevant TRs
sampled_basic_data = masked_fmri_data[trs_to_label,:]
sampled_fancy_data = fancy_masked_fmri_data[trs_to_label,:]

# create decoder
from sklearn.linear_model import LogisticRegression

my_decoder = LogisticRegression() # this has a bunch of options you can give it
train_set = np.arange(num_runs*trials_per_run/2) # first half samples
test_set = num_runs*trials_per_run/2+np.arange(num_runs*trials_per_run/2) # last half samples

# test on basic data
my_decoder.fit(sampled_basic_data[train_set],trial_labels[train_set]) # fit to training set
predictions = my_decoder.predict(sampled_basic_data[test_set]) # predict on test set
decoder_accuracy = np.mean(predictions==trial_labels[test_set]) # determine accuracy
print('basic accuracy: '+str(decoder_accuracy))

# test on fancier data
my_decoder.fit(sampled_fancy_data[train_set],trial_labels[train_set]) # fit to training set
predictions = my_decoder.predict(sampled_fancy_data[test_set]) # predict on test set
decoder_accuracy = np.mean(predictions==trial_labels[test_set]) # determine accuracy
print('fancy accuracy: '+str(decoder_accuracy))
