function pred_y = multiclassPredict(w, testX, testingClassList)

% now use test data to check according to One-vs-All methodology

% loop through testing rows
for i = 1:size(testX,1)
    % loop through trained classifiers
    for j = 1:numel(testingClassList)
        if sign(w{j}'*testX(i,:)') == -1
            pred_y_test(i,j) = 0;
        else
            pred_y_test(i,j) = sign(w{j}'*testX(i,:)');
        end
    end
    % check for ambiguities and default to previous case, if applicable
    % if ambiguity on row one, default to rest
    if sum(pred_y_test(i,:)) > 1
        if i == 1
            pred_y{i,1} = 'Rest';
        else
            pred_y(i,1) = pred_y(i-1,1);
        end
    else
        % find index of column with value of 1, if present
        col = find(pred_y_test(i,:));
        if isempty(col)
            pred_y{i,1} = 'Rest'; % default to rest, if no other class identified
        else
            pred_y{i,1} = testingClassList{col};
        end
    end
end