% Wrapper for classifier examples

close all
clearvars
rng(0)

%% Initialize variables and select/load data

% New data (12/11/18)
trainingSetDirectory = "Data\7_classes_ML_2018";
trainingFileDirectory = "Data\TrainingData_20181211-092002_092924.kdf";
trainingNeuralFeatures = 0; % 0 to exclude neural

[trainingFeatures, trainingLabels, trainingClassList] = classParser(trainingSetDirectory, trainingFileDirectory, trainingNeuralFeatures);

testingSetDirectory = "Data\7_classes_ML_2018";
testingFileDirectory = "Data\TrainingData_20181211-092002_093345.kdf";
testingNeuralFeatures = 0; % 0 to exclude neural

[testingFeatures, testingLabels, testingClassList] = classParser(testingSetDirectory, testingFileDirectory, testingNeuralFeatures);



%% Add column of ones to test data for multiplying with [bias, weights] vector
offset = ones(size(testingFeatures,1),1);
testX = horzcat(offset,testingFeatures);
testY_labels = cellstr(testingLabels);

% add column of ones to trainX for simplifying w calculations
offset = ones(size(trainingFeatures,1),1);
trainX = horzcat(offset,trainingFeatures);

%% Create cross-validation sets
numFolds = 5;
CV_indices = crossvalind('Kfold',trainingLabels,numFolds);

for fold=1:numFolds
    CV_TrainX{fold} = trainX((CV_indices ~= fold),:);
    CV_TestX{fold} = trainX((CV_indices == fold),:);
    CV_TrainY{fold} = trainingLabels(CV_indices ~= fold);
    CV_TestY{fold} = trainingLabels(CV_indices == fold);
end

%% Create One-vs-All training sets (i.e. make binary data sets)

% loop through each class and generate binary train and test sets for cross-validation
for cvSet = 1:numel(CV_TrainX)
    for i = 1:numel(trainingClassList)
        % get rows where class is current class
        idxTrain = find(strcmp(CV_TrainY{cvSet},trainingClassList{i}));
        % set positive examples of current class as 1 and all others as -1
        trainY{cvSet,i} = -1*ones(size(CV_TrainY{cvSet},1),1);
        trainY{cvSet,i}(idxTrain) = 1;
        
        idxTest = find(strcmp(CV_TestY{cvSet},trainingClassList{i}));

        testY{cvSet,i} = -1*ones(length(CV_TestY{cvSet}),1);
        testY{cvSet,i}(idxTest) = 1;
    end
end

% loop through each class and generate binary whole training set labels for
% use AFTER cross-validation
for i = 1:numel(trainingClassList)
    % get rows where class is current class
    idx = find(strcmp(trainingLabels,trainingClassList{i}));
    % set positive examlpes of current class as 1 and all others as -1
    orig_trainY{i} = -1*ones(size(trainingLabels,1),1);
    orig_trainY{i}(idx) = 1;
end

%% Simple perceptron classifier for above data

% Perform 5-fold Cross-validation to get best hyperparameters
CV_epochs = 10:20;
r_all = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001];

tic
f = waitbar(0,'Running cross-validation for Perceptron');
iter = 0;

% loop through all max epochs
for e_idx = 1:numel(CV_epochs)
    % loop through all learning rates
    for r_idx = 1:numel(r_all)
        % loop through each CV set and calculate accuracy
        for currCV = 1:size(CV_TrainX,2)
            for i = 1:numel(trainingClassList)

                iter = iter + 1;
                waitbar(iter/(numel(CV_epochs)*numel(r_all)*size(CV_TrainX,2)*numel(trainingClassList)),f,'Running cross-validation for Perceptron'); 

                w{i} = genSimplePerceptron(CV_epochs(e_idx), CV_TrainX{currCV}, trainY{currCV,i}, r_all(r_idx));

            end

            % Check accuracy against held-out data in multiclass classification
            pred_y_CV = multiclassPredict(w, CV_TestX{currCV}, testingClassList);
            count = 0;
            for acc_idx = 1:numel(pred_y_CV)
                if pred_y_CV{acc_idx,1} == CV_TestY{1,currCV}(acc_idx)
                    count = count + 1;
                end
            end
            accuracy_CV(currCV) = 100*(count/numel(CV_TestY{1,currCV}));
            clear pred_y_CV
        end
        % Calculate mean accuracy for each hyperparameter
        mean_CV_acc(r_idx, e_idx) = nanmean(accuracy_CV);
    end
end

close(f)
toc

% Decide which hyperparameter from above was best and use it to train on whole dataset
[best_tmp, row_idx] = max(mean_CV_acc);
[best_CV_acc, col_idx] = max(best_tmp);

row_idx = row_idx(col_idx);
best_r = r_all(row_idx);
best_e = CV_epochs(col_idx);

% loop through each binary training subset and generate simple perceptron
for i = 1:numel(trainingClassList)
    w{i} = genSimplePerceptron(best_e, trainX, orig_trainY{i}, best_r);
end

pred_y_test_label_best = multiclassPredict(w, testX, testingClassList);
test_accuracy_perceptron = sum(pred_y_test_label_best == testingLabels)/length(testingLabels);

% confusion matrices
conf_mat_perceptron = confusionmat(testY_labels,pred_y_test_label_best);
perceptron_conf_chart = confusionchart(conf_mat_perceptron, testingClassList);
confMatTitle = sprintf('Test Set Accuracy: %0.5f', test_accuracy_perceptron);
title({'Confusion Matrix for Perceptron', confMatTitle})

%% SVM with stochastic sub-gradient descent for above data

% hyperparameters
epochs = 5:20;
learnRates = [0.1, 0.01, 0.001, 0.0001, 0.00001];
C_params = [1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001];

tic
f = waitbar(0,'Running cross-validation for SVM');
iter=0;

% find weight vectors for individual classifiers
w = cell(numel(epochs), numel(learnRates), numel(C_params), numFolds, numel(trainingClassList));
for epoch = 1:numel(epochs)
    for learnRate = 1:numel(learnRates)
        for C_param = 1:numel(C_params)
            for cvSet = 1:numel(numFolds)
                for class = 1:numel(trainingClassList)
                    w{epoch,learnRate,C_param,cvSet,class} = genSVMStochSubgradDesc(...
                        epochs(epoch), learnRates(learnRate), C_params(C_param), CV_TrainX{cvSet}, trainY{cvSet,class});
                    iter = iter+1;
                    waitbar(iter/(numel(epochs)*numel(C_params)*numel(numFolds)*numel(learnRates)*size(CV_TrainX,2)*numel(trainingClassList)),f,'Running cross-validation for SVM');
                end
            end
        end
    end
end

close(f)
toc

% find accuracy from different hyperparams
bestAccuracy = 0;
bestEpoch = 0;
bestLR = 0;
bestC_param = 0;
for epoch = 1:numel(epochs)
    for learnRate = 1:numel(learnRates)
        for C_param = 1:numel(C_params)
            for cvSet = 1:numel(numFolds)
                pred_y_test_label{epoch,learnRate,C_param,cvSet} = multiclassPredict({w{epoch,learnRate,C_param,cvSet,:}}, CV_TestX{cvSet}, testingClassList); 
                cvSetAccuracy{cvSet} = sum(pred_y_test_label{epoch,learnRate,C_param,cvSet} == CV_TestY{cvSet})/length(CV_TestY{cvSet});
            end
            avgAccuracy{epoch,learnRate,C_param} = mean(cvSetAccuracy{:});
            if avgAccuracy{epoch,learnRate,C_param} > bestAccuracy
                bestAccuracy = avgAccuracy{epoch,learnRate,C_param};
                bestEpoch = epochs(epoch);
                bestLR = learnRates(learnRate);
                bestC_param = C_params(C_param);
            end
        end
    end
end
fprintf('Best Learning Rate: %0.0e\nBest Accuracy: %0.5f\nBest Epochs: %i\nBest C-param: %0.0e\n',...
    bestLR, bestAccuracy, bestEpoch, bestC_param);


% now train on all training data and test on test set
for i=1:numel(trainingClassList)
    w_best{i} = genSVMStochSubgradDesc(bestEpoch, bestLR, bestC_param, trainingFeatures, orig_trainY{i});
end
pred_y_test_label_best = multiclassPredict(w_best, testingFeatures, testingClassList); 
test_accuracy_svm = sum(pred_y_test_label_best == testingLabels)/length(testingLabels);

% confusion matrices
conf_mat_svm = confusionmat(testY_labels,pred_y_test_label_best);
svm_conf_chart = confusionchart(conf_mat_svm, testingClassList);
confMatTitle = sprintf('Test Set Accuracy: %0.5f', test_accuracy_svm);
title({'Confusion Matrix for SVM', confMatTitle})

%% Logistic Regression with stochastic gradient descent for above data

% Perform 5-fold Cross-validation to get best hyperparameters
CV_epochs = 10:20;
r_init = [0.001, 0.0001, 0.00001];
sig_all = [0.01, 0.1, 1, 10, 100, 1000, 10000];

tic
f = waitbar(0,'Running cross-validation for Logistic Regression');
iter = 0;

% loop through all max epochs
for e_idx = 1:numel(CV_epochs)
    % loop through all regularization terms
    for s_idx = 1:numel(sig_all)
        % loop through all learning rates
        for r_idx = 1:numel(r_init)
            % loop through each CV set and calculate accuracy
            for currCV = 1:size(CV_TrainX,2)
                for i = 1:numel(trainingClassList)

                    iter = iter + 1;
                    waitbar(iter/(numel(CV_epochs)*numel(sig_all)*numel(r_init)*size(CV_TrainX,2)*numel(trainingClassList)),f,'Running cross-validation for Logistic Regression'); 

                    w{i} = genLogReg(CV_epochs(e_idx), CV_TrainX{currCV}, trainY{currCV,i}, r_init(r_idx), sig_all(s_idx));

                end

                % Check accuracy against held-out data in multiclass classification
                pred_y_CV = multiclassPredict(w, CV_TestX{currCV}, testingClassList);
                count = 0;
                for acc_idx = 1:numel(pred_y_CV)
                    if pred_y_CV{acc_idx,1} == CV_TestY{1,currCV}(acc_idx)
                        count = count + 1;
                    end
                end
                accuracy_CV(currCV) = 100*(count/numel(CV_TestY{1,currCV}));
                clear pred_y_CV
            end
            % Calculate mean accuracy for each hyperparameter combo
            mean_CV_acc(r_idx,s_idx,e_idx) = nanmean(accuracy_CV);
        end
    end
end

close(f)
toc

% Decide which hyperparameter from above was best and use it to train on whole dataset
[best_tmp1, row_idx] = max(mean_CV_acc);
[best_tmp2, col_idx] = max(best_tmp1);
[best_CV_acc, dim_3_idx] = max(best_tmp2);
col_idx = col_idx(dim_3_idx);
row_idx = row_idx(col_idx);

best_r = r_init(row_idx);
best_sig = sig_all(col_idx);
best_e = CV_epochs(dim_3_idx);

% loop through each binary training subset and generate logistic regression classifier
for i = 1:numel(trainingClassList)
    w{i} = genLogReg(best_e, trainX, orig_trainY{i}, best_r, best_sig);
end

pred_y_test_label_best = multiclassPredict(w, testX, testingClassList);
test_accuracy_logistic = sum(pred_y_test_label_best == testingLabels)/length(testingLabels);

% confusion matrices
conf_mat_logistic = confusionmat(testY_labels,pred_y_test_label_best);
logistic_conf_chart = confusionchart(conf_mat_logistic, testingClassList);
confMatTitle = sprintf('Test Set Accuracy: %0.5f', test_accuracy_logistic);
title({'Confusion Matrix for Logistic Regression', confMatTitle})
