% function mi = mutual_information(X, Y)
% 
% % Get the number of samples
% n_samples = length(X);
% 
% % Compute the joint probability distribution P(X,Y)
% joint_counts = accumarray([X(:), Y(:)], 1);
% joint_prob = joint_counts / n_samples;
% 
% % Compute the marginal probability distributions P(X) and P(Y)
% prob_X = sum(joint_prob, 2);
% prob_Y = sum(joint_prob, 1);
% 
% % Compute the mutual information
% mi = 0;
% for i = 1:size(joint_prob, 1)
%     for j = 1:size(joint_prob, 2)
%         if joint_prob(i, j) > 0 && prob_X(i) > 0 && prob_Y(j) > 0
%             mi = mi + joint_prob(i, j) * log2(joint_prob(i, j) / (prob_X(i) * prob_Y(j)));
%         end
%     end
% end
% 
% end
function MI = mutual_information(x, y, numBins)

    % Ensure x and y are column vectors
    x = x(:);
    y = y(:);

    % Get lengths
    len_x = length(x);
    len_y = length(y);

    % If lengths of x and y are not equal, truncate or pad shorter vector with its mean
    if len_x > len_y
        x = x(1:len_y);
    elseif len_y > len_x
        y = y(1:len_x);
    end

    % Discretize x and y using histogram
    [~, ~, x] = histcounts(x, numBins);
    [~, ~, y] = histcounts(y, numBins);

    % Calculate joint histogram
    jointHist = histcounts2(x, y, numBins);

    % Normalize joint histogram to get joint probability mass function
    jointProb = jointHist / numel(x);

    % Marginal probabilities
    x_prob = sum(jointProb, 2);
    y_prob = sum(jointProb, 1);

    % Calculate mutual information
    MI = 0;
    for i = 1:numBins
        for j = 1:numBins
            if jointProb(i,j) > 0
                MI = MI + jointProb(i,j) * log2(jointProb(i,j) / (x_prob(i) * y_prob(j)));
            end
        end
    end
end
