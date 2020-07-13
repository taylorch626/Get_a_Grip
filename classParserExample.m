trainingSetDirectory = "C:\Users\Administrator\Code\Tasks\FeedbackDecode\resources\TrainingSets\Classifier\7_classes_ML_2018";
trainingFileDirectory = "\\PNILABVIEW\PNILabview_R6\Data\Lab_sEMG\20181105-110934\TrainingData_20181105-110934_111501.kdf"; % my data
neuralFeatures = 0; % 0 to exclude neural

[features, labels, classList] = classParser(trainingSetDirectory, trainingFileDirectory, neuralFeatures);