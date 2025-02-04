function [A, B, U, V] = CCA(X, Y)
    % Inputs:
    % X - Matrix of signals (n_samples x n_features)
    % Y - Matrix of reference signals (n_samples x n_features)
    % Outputs:
    % A, B - Projection matrices for X and Y
    % U, V - Canonical variables

    % Subtract mean to center the data
    X = bsxfun(@minus, X, mean(X));
    Y = bsxfun(@minus, Y, mean(Y));
    
    % Compute covariance matrices
    Cxx = cov(X);
    Cyy = cov(Y);
    Cxy = cov(X, Y); % cross-covariance matrix

    % Solve the generalized eigenvalue problem
    [A, D] = eig(Cxy' * (inv(Cxx) * Cxy), Cyy);
    [B, ~] = eig(Cxy * (inv(Cyy) * Cxy'), Cxx);

    % Sort eigenvalues and eigenvectors
    [~, idx] = sort(diag(D), 'descend');
    A = A(:, idx);
    B = B(:, idx);

    % Compute canonical variables
    U = X * A;
    V = Y * B;
end
