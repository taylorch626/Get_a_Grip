function [XAligned, ZAligned] = alignTrainingData(XOrig, ZOrig, badIdxs, method)
% inputs
%   Xorig: 12 x n samples kinematic data movement cue
%   Zorig: 720 x n samples Feature data
%   BadChans: vector of bad chan idx
%   method: string 'standard' (correlation based) or 'trialByTrial' 
% outputs:
% XAligned: 12 x n aligned kinematics
% ZAligned: 720 x n aligned features
% smw

% init
XAligned = XOrig;
ZAligned = ZOrig;
disp('Data alignment...');

TrainZ(badIdxs, :) = 0;
switch method
    case 'standard'
        % find lag, apply to training data
        [Mvnts,Idxs,MaxLag] = autoSelectMvntsChsCorr_FD(XOrig,ZOrig,0.4,badIdxs);
        ZAligned = circshift(ZOrig, MaxLag,2);
    case 'trialByTrial'
        [XAligned,ZAligned] = realignIterCombo(XOrig,ZOrig);%
        % note: could zap badKalmanIdxs at this point before sending to train
        
end