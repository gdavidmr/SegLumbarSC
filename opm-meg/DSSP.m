% function [denoised_data] = DSSP(data, S, num_components)
% % DSSP: Dual Signal Subspace Projection
% % Inputs:
% %   data: Input data matrix (channels x time points)
% %   S: Spatial basis matrix (channels x spatial components)
% %   num_components: Number of interference components to remove
% % Output:
% %   denoised_data: Denoised data matrix
% 
% % Step 1: Project data onto signal subspace and its complement
% P_s = S * pinv(S);  % Projection matrix onto signal subspace
% P_n = eye(size(P_s)) - P_s;  % Projection matrix onto noise subspace
% 
% X_s = P_s * data;  % Data projected onto signal subspace
% X_n = P_n * data;  % Data projected onto noise subspace
% 
% % Step 2: Perform SVD on the projected data matrices
% [U_s, ~, ~] = svd(X_s, 'econ');
% [U_n, ~, ~] = svd(X_n, 'econ');
% 
% % Step 3: Estimate interference subspace
% U_i = U_s(:, 1:num_components);
% V_i = U_n(:, 1:num_components);
% 
% % Step 4: Construct interference projection matrix
% P_i = [U_i V_i] * pinv([U_i V_i]);
% 
% % Step 5: Remove interference by projecting data onto complement of interference subspace
% denoised_data = (eye(size(P_i)) - P_i) * data;
% end


function [denoised_data] = DSSP(data, S, num_components)
    % Project data onto signal subspace and its complement
    P_s = S * pinv(S);
    P_n = eye(size(P_s)) - P_s;
    X_s = P_s * data;
    X_n = P_n * data;

    % Perform SVD on projected data matrices
    [U_s, ~, ~] = svd(X_s, 'econ');
    [U_n, ~, ~] = svd(X_n, 'econ');

    % Estimate interference subspace
    U_i = U_s(:, 1:num_components);
    V_i = U_n(:, 1:num_components);

    % Construct interference projection matrix
    P_i = [U_i V_i] * pinv([U_i V_i]);

    % Remove interference
    denoised_data = (eye(size(P_i)) - P_i) * data;
end
