addpath(genpath(pwd));
load('finger_data.mat');

% need to convert raw int32's from the MRI scanner
% to a format Matlab likes (in this case, 'double')
fmri_data = double(fmri_data);

% create an empty dataset to output to
processed_data = zeros(size(fmri_data));

for run_num = 0:max(runs)
    % extract data from each run to process separately
    run_data = fmri_data(runs==run_num,:);

    % let's try z-scoring the data
    processed_run_data = zscore(run_data);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % add more processing here %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % put the processed data into the final dataset here
    processed_data(runs==run_num,:) = processed_run_data;
end

% extract just the finger movement data, without baseline period
processed_data = processed_data(labels~=-1,:);
labels = labels(labels~=-1);

% now extract the TR we think will give the best decoding
TRS_PER_TRIAL = 8; % there are 8 TRs in each run
DECODING_TR = 6; % decode from the 6th TR only
trs_to_extract = DECODING_TR:TRS_PER_TRIAL:size(processed_data,1);
extracted_processed_data = processed_data(trs_to_extract,:);
extracted_labels = labels(trs_to_extract);

% now do decoding; first, separate data in training and test sets
% the training set is what we use to build our decoder,
% and the test set is what we use to see how accurate the decoder is
[train_examples,test_examples] = separate_train_test(extracted_labels, 0.5);

% train a multinomial regularized logistic regression classifier, 1-vs-rest
% this is the fastest decoder to train from this toolbox
[clf_weights, ix_eff, errTable_tr, errTable_te] = muclsfy_rlrvarovrm(...
    extracted_processed_data(train_examples,:), extracted_labels(train_examples),...
    extracted_processed_data(test_examples,:), extracted_labels(test_examples),...
    'nlearn',100,'nstep',25);

% after training, it will give you an output accuracy (e.g. 65%)
% but you can also examine the errors it made. For example:
disp(errTable_te) % shows you how well the decoder did on each finger
