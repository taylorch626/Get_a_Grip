function w = genSimplePerceptron(epochs, trainX, trainY, r)

% Set an arbitrary learning rate, r <-- this should be optimized via CV at
% some point
% r = 0.01;

% Initialize an empty weight vector w/ included bias
% the bias will be the first element in w
w = -0.01 + (1/50)*rand(size(trainX,2), 1); % bound between -0.01 and 0.01

% Initialize update count
% countUpdate = 0;

for e = 1:epochs % loop for number of epochs specified at top

    % shuffle the data during each epoch
    new_idx = randperm(size(trainX,1));
    currX = trainX(new_idx,:);
    currY = trainY(new_idx);

    for i = 1:size(trainX,1)
        % make a prediction
        pred_y = sign(w'*currX(i,:)'); % should be a scalar
        if pred_y == 0
            pred_y = 1;
        end
        % check if prediction requires an update
        if currY(i)*pred_y <= 0
            w = w + r*currY(i)*currX(i,:)'; % update weights and bias
%             countUpdate = countUpdate + 1; % update counter for # of updates
        end
    end
    
    % keep track of w at each epoch
%     w_hist(:,e) = w;
    
%     % test accuracy of classifier on development set at end of each epoch
%     pred_y_dev = sign(w'*devX');
%     num_mistakes_dev = sum(abs(pred_y_dev' - devY))/2; % divide by 2 to account for false being -1
%     accuracy_dev(e) = 100*((numel(devY) - num_mistakes_dev) / numel(devY));
end