import os # basic file and path operations
import numpy as np # math and arrays
from nilearn.image import load_img # basic image loading
from nilearn.input_data import NiftiMasker # image masking and more!
from scipy import signal
from scipy.signal import lfilter
from scipy.stats import zscore
from scipy.stats import sem
from scipy.signal import savgol_filter
import pandas as pd # great tool for playing with spreadsheet-type data
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt # for plotting

# set up basic directories
data_dir = 'fmri' # directory where subj_id folder exists
subj_id = 'ff002' # subject ID in directory
sess_id = 'sess1' # session ID in BOLD subdirectory
bold_dir = os.path.join(data_dir,subj_id,'bold',sess_id)
ref_dir = os.path.join(data_dir,subj_id,'ref')

# select and load mask
mask_id = 'mask_lh_s1'
mask_file = os.path.join(ref_dir,mask_id+'.nii')
mask_img = load_img(mask_file)

# load raw data
raw_fmri_data = load_img(os.path.join(bold_dir,'*.nii'), dtype=np.float32)

# mask the data
masker = NiftiMasker(mask_img=mask_img)
masker.fit(raw_fmri_data)
masked_fmri_data = masker.transform(raw_fmri_data)


def fmri_filter(masked_data, number_runs, trs_in_run, base_trs, max_frame, sg, dt, zs):
    processed_real_time_data = np.zeros((masked_data.shape))
    filter_len = 3
    b = np.ones(filter_len)/filter_len
    a = np.ones(1)
    for r in range(number_runs):
        temp_data = masked_data[np.where(run_labels==r)]
        if zs ==1:
            for n in range(trs_in_run):
                real_time_temp = temp_data[0:n+1,:]
                if n==base_trs-1:
                    processed = real_time_temp
                    processed = zscore(processed)
                    processed_real_time_data[trs_per_run*r:trs_per_run*r+n+1,:] = processed[0:n+1,:]
                if n>base_trs-1:
                    processed = zscore(real_time_temp)
                    processed_real_time_data[trs_per_run*r+n,:] = processed[n,:]
        if dt == 1:
            for n in range(trs_in_run):
                real_time_temp = temp_data[0:n+1,:]
                if n==base_trs-1:
                    processed = real_time_temp
                    processed = zscore(processed)
                    processed_real_time_data[trs_per_run*r:trs_per_run*r+n+1,:] = processed[0:n+1,:]
                if n>base_trs-1:
                    processed = signal.detrend(real_time_temp)
                    processed = lfilter(b, a, processed)
                    processed = zscore(processed)
                    processed_real_time_data[trs_per_run*r+n,:] = processed[n,:]
        elif sg == 1:
            for n in range(trs_in_run): # in range(len(temp_data))
                real_time_temp = temp_data[0:n+1,:]
                if n==base_trs-1:
                    processed = real_time_temp
                    processed = zscore(processed)
                    processed_real_time_data[trs_per_run*r:trs_per_run*r+n+1,:] = processed[0:n+1,:]
                if n>base_trs-1 & n<max_frame:
                    if n%2 == 1:
                        framelength = n
                    else:
                        framelength = n-1
                    sg_filter = savgol_filter(real_time_temp, framelength, 2)
                    processed = real_time_temp - sg_filter
                    processed = lfilter(b, a, processed)
                    processed = zscore(processed)
                    processed_real_time_data[trs_per_run*r+n,:] = processed[n,:]
                if n>base_trs-1 & n >= max_frame:
                    framelength = max_frame
                    sg_filter = savgol_filter(real_time_temp, framelength, 2)
                    processed = real_time_temp - sg_filter
                    processed = lfilter(b, a, processed)
                    processed = zscore(processed)
                    processed_real_time_data[trs_per_run*r+n,:] = processed[n,:]
    return processed_real_time_data

num_runs = 8; baseline_trs = 20; trials_per_run = 20; trs_per_trial = 8;
trs_per_run = baseline_trs + trials_per_run*trs_per_trial
run_labels = np.repeat(range(num_runs),trs_per_run)

max_framelength = 61


real_time_sg = fmri_filter(masked_fmri_data, num_runs, trs_per_run, baseline_trs, max_framelength, 1, 0, 0) #sg filter

real_time_dt = fmri_filter(masked_fmri_data, num_runs, trs_per_run, baseline_trs, max_framelength, 0, 1, 0) #dt filter

real_time_zs = fmri_filter(masked_fmri_data, num_runs, trs_per_run, baseline_trs, max_framelength, 0, 0, 1) #zs filter

#load behavioral data
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
sampled_zs_data = real_time_zs[trs_to_label,:]
sampled_sg_data = real_time_sg[trs_to_label,:]
sampled_dt_data = real_time_dt[trs_to_label,:]

#create decoder
my_decoder = LogisticRegression(multi_class='ovr', solver='liblinear')   # this has a bunch of options you can give it
kf = KFold(n_splits = 8)
X = np.arange(num_runs*trials_per_run)
decoder_accuracy_sg = np.zeros(trs_per_trial)
it = -1
for train_set, test_set in kf.split(X):
    it = it+1
    my_decoder.fit(sampled_sg_data[train_set],trial_labels[train_set]) # fit to training set
    predictions = my_decoder.predict(sampled_sg_data[test_set]) # predict on test set
    decoder_accuracy = np.mean(predictions==trial_labels[test_set]) # determine accuracy
    decoder_accuracy_sg[it] = decoder_accuracy
#print('real time sg accuracy: '+str(decoder_accuracy_sg))
print('real time sg accuracies: ' +str(decoder_accuracy_sg))

decoder_accuracy_dt = np.zeros(trs_per_trial)
it = -1
for train_set, test_set in kf.split(X):
    it = it+1
    my_decoder.fit(sampled_dt_data[train_set],trial_labels[train_set]) # fit to training set
    predictions = my_decoder.predict(sampled_dt_data[test_set]) # predict on test set
    decoder_accuracy = np.mean(predictions==trial_labels[test_set]) # determine accuracy
    decoder_accuracy_dt[it] = decoder_accuracy
print('real time dt accuracies: '+str(decoder_accuracy_dt))

decoder_accuracy_zs = np.zeros(trs_per_trial)
it = -1
for train_set, test_set in kf.split(X):
    it = it+1
    my_decoder.fit(sampled_dt_data[train_set],trial_labels[train_set]) # fit to training set
    predictions = my_decoder.predict(sampled_zs_data[test_set]) # predict on test set
    decoder_accuracy = np.mean(predictions==trial_labels[test_set]) # determine accuracy
    decoder_accuracy_zs[it] = decoder_accuracy
print('real time zs accuracies: '+str(decoder_accuracy_zs))


#pick a voxel
abs_coef = np.absolute(my_decoder.coef_)
sum_abs_coef = np.sum(abs_coef,axis=0)
sample_voxel = np.argmax(sum_abs_coef) # random voxel index - once we get to decoding, pick an important one!

#plot
plt.ion() # this makes your plots appear immediately, otherwise need to call plt.show() after plt.plot()
real_time_sample_zs_voxel_activity = sampled_zs_data[:,sample_voxel]
real_time_sample_sg_voxel_activity = sampled_sg_data[:,sample_voxel] # rows are time, columns are voxels
real_time_sample_dt_voxel_activity = sampled_dt_data[:,sample_voxel]
plt.plot(real_time_sample_sg_voxel_activity)
plt.plot(real_time_sample_dt_voxel_activity)
plt.plot(real_time_sample_zs_voxel_activity)
plt.xlabel('TR')
plt.ylabel('voxel activity')
plt.legend(['sg', 'dt', 'zs'])
plt.title('real-time fmri filtering')

#bar plot
x_axis = ['sg', 'dt', 'zs']
sg_mean = np.mean(decoder_accuracy_sg)
sg_sem = sem(decoder_accuracy_sg)
dt_mean = np.mean(decoder_accuracy_dt)
dt_sem = sem(decoder_accuracy_dt)
zs_mean = np.mean(decoder_accuracy_zs)
zs_sem = sem(decoder_accuracy_zs)

y_axis = [sg_mean, dt_mean, zs_mean]
st_err = [sg_sem, dt_sem, zs_sem]
ind = np.arange(len(x_axis))
plt.bar(ind, y_axis, yerr=st_err)
plt.xticks(ind,x_axis)
plt.ylabel('Decoding Accuracy')
plt.title('real-time fmri decoding accuracies')

#pandas dataframe
sub = np.array([subj_id])
sub = np.repeat(sub,3)
filt = ['sg', 'dt', 'zs']
acc = [sg_mean, dt_mean, zs_mean]
error = [sg_sem, dt_sem, zs_sem]
d = {'subject': sub, 'filter': filt, 'accuracy': acc, 'std err': error}
pd.DataFrame(data=d).to_csv(sub[0]+'.csv')
