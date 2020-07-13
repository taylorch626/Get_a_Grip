function [features, labels, classList] = classParser(trainingSetDirectory, trainingFileDirectory, includeNeural, varargin)
% Takes a training file and labels the features based on the kinematics
% values in the training file directory.
% 
% Inputs:
%     trainingSetDirectory: directory of training set
%     trainingFileDirectory: directory of training KDF
%     includeNeural: 1 to include neural features
% 
% Optional Inputs:
%     kinematicThreshold: defaults to 0.9 - decimal of threshold for
%         kinematic to be labeled as a class, e.g., 0.9 labels training data
%         that is 90% of the class (i.e., includes some rise time)
%
% Outputs:
%     labeledData: cell array of classes with their training data
%     classes: cell array of class names derived from trainingSetDirectory
% 
% MP 11/2018

if nargin < 4
    kinematicThreshold = 0.9;
else
    kinematicThreshold = varargin{1};
end

checkClassSeparation = 0; % makes a square matrix to show if classes are being separated properly

%% get list of classes and kinematic values
% classes
delimiter = {',',';'};
formatSpecText = '%s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%[^\n\r]';
fileID = fopen(trainingSetDirectory, 'r');
dataArrayText = textscan(fileID, formatSpecText, 'Delimiter', delimiter, 'TextType', 'string', 'HeaderLines', 0, 'ReturnOnError', false, 'EndOfLine', '\r\n');
classList = cellstr(dataArrayText{1});
fclose(fileID);

% kinematics
fileID = fopen(trainingSetDirectory, 'r');
startRow = 1;
formatSpecKin = '%*s%*s%f%*s%*s%*s%f%*s%*s%*s%f%*s%*s%*s%f%*s%*s%*s%f%*s%*s%*s%f%*s%*s%*s%f%*s%*s%*s%f%*s%*s%*s%f%*s%*s%*s%f%*s%*s%*s%f%*s%*s%*s%f%*s%*s%*s%[^\n\r]';
dataArrayKin = textscan(fileID, formatSpecKin, 'Delimiter', delimiter, 'TextType', 'string', 'HeaderLines', 0, 'ReturnOnError', false, 'EndOfLine', '\r\n');
for block=2:length(startRow)
    frewind(fileID);
    dataArrayBlock = textscan(fileID, formatSpec, endRow(block)-startRow(block)+1, 'Delimiter', delimiter, 'TextType', 'string', 'HeaderLines', startRow(block)-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
    for col=1:length(dataArrayKin)
        dataArrayKin{col} = [dataArrayKin{col};dataArrayBlock{col}];
    end
end
fclose(fileID);
classesMatrix = [dataArrayKin{1:end-1}]';
classKinematics = num2cell(classesMatrix,1);

%% apply labels to data
[Kinematics,Features,~,~,~] = readKDF(trainingFileDirectory);
if(includeNeural)
	badIdxs = [];
else
    badIdxs = [1:192]; % ignore neural (ch 1:192) use emg only
end
alignMethod = 'standard'; % 'trialByTrial' or 'standard'
[Kinematics, Features] = alignTrainingData(Kinematics, Features, badIdxs, alignMethod);

if(checkClassSeparation)
    class = 1:7;
    index = [60, 1500, 650, 3090, 460, 870, 2080];
    diffMat = zeros(length(class),length(index));
    precisionTolerance = 1e-5;
    for classInd=1:length(classList)
        for kinInd=1:length(index)
            if (sum(sign(Kinematics(:,index(kinInd))) == sign(classKinematics{classInd}))) == 12
                if (sum(abs(Kinematics(:,index(kinInd))) >= kinematicThreshold*abs(classKinematics{classInd})) == 12) && (sum(abs(Kinematics(:,index(kinInd))) <= precisionTolerance+abs(classKinematics{classInd})) == 12)
                    diffMat(classInd,kinInd) = 1;
                end
            end
        end
    end
end

precisionTolerance = 1e-5;
labeledIndex = cell(length(classList),2);
for kinInd=1:size(Kinematics,2)
    count = 0;
    for classInd=1:length(classList)
        % check that all kinematics move same direction
        if sum(sign(Kinematics(:,kinInd)) == sign(classKinematics{classInd})) == 12
            % check that abs val of kinematics are greater than threshold
            % and less than full kinematics of class + tolerance for
            % machine precision
        	if (sum(abs(Kinematics(:,kinInd)) >= kinematicThreshold*abs(classKinematics{classInd})) == 12)...
                    && (sum(abs(Kinematics(:,kinInd)) <= precisionTolerance+abs(classKinematics{classInd})) == 12)
                labeledIndex{classInd,1} = [labeledIndex{classInd,1} kinInd];
                labeledIndex{classInd,2} = [labeledIndex{classInd,2} Features(:,kinInd)];
                count = count + 1;
            end
        end
    end
    if count > 1
        warning(['Features used for duplicate classes on Kinematics index: ' num2str(kinInd)]);
    end
end
    
%% Mod for Taylor
% m x n data, m examples, n features
% m x 1 column of labels
% features = double
features = [];
labels = strings;
for classInd=1:size(labeledIndex,1)
    features = [features; labeledIndex{classInd,2}'];
    for label = 1:length(labeledIndex{classInd})
        labels = [labels; classList{classInd}];
    end
end
labels = labels(2:end);