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
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt # for plotting

def fmri_filter(masked_data, num_runs, trs_per_run, base_trs, max_frame, sg, dt, zs):
    run_labels = np.repeat(range(num_runs),trs_per_run)
    processed_real_time_data = np.zeros((masked_data.shape))
    filter_len = 3
    b = np.ones(filter_len)/filter_len
    a = np.ones(1)
    for r in range(num_runs):
        temp_data = masked_data[np.where(run_labels==r)]
        if zs == 1:
            for n in range(trs_per_run):
                real_time_temp = temp_data[0:n+1,:]
                if n==base_trs-1:
                    processed = real_time_temp
                    processed = zscore(processed)
                    processed_real_time_data[trs_per_run*r:trs_per_run*r+n+1,:] = processed[0:n+1,:]
                if n>base_trs-1:
                    processed = zscore(real_time_temp)
                    processed_real_time_data[trs_per_run*r+n,:] = processed[n,:]
        if dt == 1:
            for n in range(trs_per_run):
                real_time_temp = temp_data[0:n+1,:]
                if n==base_trs-1:
                    processed = real_time_temp
                    processed = zscore(processed)
                    processed_real_time_data[trs_per_run*r:trs_per_run*r+n+1,:] = processed[0:n+1,:]
                if n>base_trs-1:
                    processed = signal.detrend(real_time_temp, axis=0)
                    processed = lfilter(b, a, processed, axis=0)
                    processed = zscore(processed)
                    processed_real_time_data[trs_per_run*r+n,:] = processed[n,:]
        elif sg == 1:
            for n in range(trs_per_run): # in range(len(temp_data))
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
                    sg_filter = savgol_filter(real_time_temp, framelength, 2, axis=0)
                    processed = real_time_temp - sg_filter
                    processed = lfilter(b, a, processed, axis=0)
                    processed = zscore(processed)
                    processed_real_time_data[trs_per_run*r+n,:] = processed[n,:]
                if n>base_trs-1 & n >= max_frame:
                    framelength = max_frame
                    sg_filter = savgol_filter(real_time_temp, framelength, 2, axis=0)
                    processed = real_time_temp - sg_filter
                    processed = lfilter(b, a, processed, axis=0)
                    processed = zscore(processed)
                    processed_real_time_data[trs_per_run*r+n,:] = processed[n,:]
    return processed_real_time_data


def analyze_subject(subj_id='ff001'):
    # set up basic directories
    sess_id = 'sess1'
    data_dir = '/Users/eo5629/fmri' # directory where subj_id folder exists
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
    num_runs = 8; baseline_trs = 20; trials_per_run = 20; trs_per_trial = 8;
    trs_per_run = baseline_trs + trials_per_run*trs_per_trial

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
    my_decoder = LogisticRegression(multi_class='ovr', solver='lbfgs') # this has a bunch of options you can give it
    kf = KFold(n_splits = 8)
    feature_selector = SelectKBest(k=100)
    X = np.arange(num_runs*trials_per_run)

    decoder_accuracy_sg = np.zeros(num_runs)
    it = -1
    for train_set, test_set in kf.split(X):
        it = it+1
        # my_decoder.fit(sampled_sg_data[train_set],trial_labels[train_set]) # fit to training set
        feature_selector.fit(sampled_sg_data[train_set],trial_labels[train_set])
        my_decoder.fit(feature_selector.transform(sampled_sg_data[train_set]),trial_labels[train_set]) # fit to training set
        predictions = my_decoder.predict(feature_selector.transform(sampled_sg_data[test_set])) # predict on test set
        decoder_accuracy = np.mean(predictions==trial_labels[test_set]) # determine accuracy
        decoder_accuracy_sg[it] = decoder_accuracy

    decoder_accuracy_dt = np.zeros(num_runs)
    it = -1
    for train_set, test_set in kf.split(X):
        it = it+1
        # my_decoder.fit(sampled_dt_data[train_set],trial_labels[train_set]) # fit to training set
        feature_selector.fit(sampled_dt_data[train_set],trial_labels[train_set])
        my_decoder.fit(feature_selector.transform(sampled_dt_data[train_set]),trial_labels[train_set]) # fit to training set
        predictions = my_decoder.predict(feature_selector.transform(sampled_dt_data[test_set])) # predict on test set
        decoder_accuracy = np.mean(predictions==trial_labels[test_set]) # determine accuracy
        decoder_accuracy_dt[it] = decoder_accuracy

    decoder_accuracy_zs = np.zeros(num_runs)
    it = -1
    for train_set, test_set in kf.split(X):
        it = it+1
        # my_decoder.fit(sampled_dt_data[train_set],trial_labels[train_set]) # fit to training set
        feature_selector.fit(sampled_zs_data[train_set],trial_labels[train_set])
        my_decoder.fit(feature_selector.transform(sampled_zs_data[train_set]),trial_labels[train_set]) # fit to training set
        predictions = my_decoder.predict(feature_selector.transform(sampled_zs_data[test_set])) # predict on test set
        decoder_accuracy = np.mean(predictions==trial_labels[test_set]) # determine accuracy
        decoder_accuracy_zs[it] = decoder_accuracy

    # take means and SEMs
    sg_mean = np.mean(decoder_accuracy_sg)
    sg_sem = sem(decoder_accuracy_sg)
    dt_mean = np.mean(decoder_accuracy_dt)
    dt_sem = sem(decoder_accuracy_dt)
    zs_mean = np.mean(decoder_accuracy_zs)
    zs_sem = sem(decoder_accuracy_zs)

    #pandas dataframe
    sub = np.array([subj_id])
    sub = np.repeat(sub,3)
    filt = ['sg', 'dt', 'zs']
    acc = [sg_mean, dt_mean, zs_mean]
    error = [sg_sem, dt_sem, zs_sem]
    d = {'subject': sub, 'filter': filt, 'accuracy': acc, 'std err': error}
    pd.DataFrame(data=d).to_csv('results/'+sub[0]+'.csv')
    print 'done '+ subj_id

def analyze_multi_subject(subj_ids=['ff001','ff002','ff003','ff004','ff005','ff006']):
    for subj_id in subj_ids:
        analyze_subject(subj_id)
