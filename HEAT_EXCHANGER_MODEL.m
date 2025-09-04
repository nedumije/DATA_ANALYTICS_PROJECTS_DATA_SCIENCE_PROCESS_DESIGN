% This script performs a comparative analysis of three machine learning
% algorithms for predictive modeling on the heat exchanger dataset.
% The models compared are:
% 1. Linear Regression
% 2. Ridge Regression
% 3. K-Nearest Neighbors (KNN) Regression

% Clear the workspace and command window for a clean start.
clear;
clc;

% =========================================================================
% HELPER FUNCTION
% =========================================================================
% This function calculates and returns the evaluation metrics.
function [MSE, RMSE, R2] = evaluate_model(y_actual, y_predicted)
    m = length(y_actual);

    % Calculate Mean Squared Error (MSE).
    MSE = (1/m) * sum((y_predicted - y_actual).^2);

    % Calculate Root Mean Squared Error (RMSE).
    RMSE = sqrt(MSE);

    % Calculate R-squared ($R^2$).
    SS_tot = sum((y_actual - mean(y_actual)).^2);
    SS_res = sum((y_actual - y_predicted).^2);
    R2 = 1 - (SS_res / SS_tot);
end

% =========================================================================
% 1. DATA IMPORT AND ROBUST PREPROCESSING
% =========================================================================

try
    fprintf('Loading and preprocessing the dataset...\n');

    % Read the header to identify the columns by name.
    header_fid = fopen('heat_exchanger_network_optimization_dataset.csv', 'r');
    header_line = fgetl(header_fid);
    fclose(header_fid);
    column_names = strsplit(header_line, ',');

    % Find the indices for our target variable and features.
    target_idx = find(strcmp(column_names, 'utility_cost'));
    feature_names = {'heat_recovered', 'utility_usage', 'heat_exchanger_area', 'heat_exchanger_efficiency', 'pinch_violation'};
    feature_idxs = [];
    for i = 1:length(feature_names)
        idx = find(strcmp(column_names, feature_names{i}));
        if ~isempty(idx)
            feature_idxs = [feature_idxs, idx];
        else
            error('Column "%s" not found in the dataset.', feature_names{i});
        end
    end

    % Load the numeric data, skipping the header row.
    data = dlmread('heat_exchanger_network_optimization_dataset.csv', ',', 1, 0);

    % Define the feature matrix X and the target vector y.
    X = data(:, feature_idxs);
    y = data(:, target_idx);

    fprintf('Dataset loaded and features selected successfully.\n');

    % =========================================================================
    % 2. DATA SPLITTING AND FEATURE SCALING
    % =========================================================================

    % Split the data into training (70%) and testing (30%) sets.
    train_ratio = 0.7;
    m = size(X, 1);
    rand_indices = randperm(m);
    train_size = floor(train_ratio * m);

    X_train = X(rand_indices(1:train_size), :);
    y_train = y(rand_indices(1:train_size), :);

    X_test = X(rand_indices(train_size+1:end), :);
    y_test = y(rand_indices(train_size+1:end), :);

    % Feature scaling is applied to normalize the data, which is crucial
    % for algorithms like linear regression and KNN.
    fprintf('\nScaling features...\n');
    X_max = max(X_train);
    X_min = min(X_train);

    % Apply scaling to the training set.
    X_train_scaled = (X_train - repmat(X_min, size(X_train, 1), 1)) ./ (repmat(X_max - X_min, size(X_train, 1), 1));

    % Apply the SAME scaling parameters to the test set to avoid data leakage.
    X_test_scaled = (X_test - repmat(X_min, size(X_test, 1), 1)) ./ (repmat(X_max - X_min, size(X_test, 1), 1));

    % Add the intercept term to the scaled linear model feature matrices.
    X_train_lin = [ones(size(X_train_scaled, 1), 1), X_train_scaled];
    X_test_lin = [ones(size(X_test_scaled, 1), 1), X_test_scaled];

    fprintf('Data split and scaling complete.\n');

    % =========================================================================
    % 3. MODEL TRAINING AND EVALUATION
    % =========================================================================

    % Initialize a structure to store the results for comparison.
    model_results = {};

    % --- Model 1: Linear Regression ---
    fprintf('\n--- Training Linear Regression Model ---\n');
    theta_lin = pinv(X_train_lin' * X_train_lin) * X_train_lin' * y_train;
    y_pred_lin = X_test_lin * theta_lin;
    [MSE_lin, RMSE_lin, R2_lin] = evaluate_model(y_test, y_pred_lin);
    model_results{1} = struct('name', 'Linear Regression', 'predictions', y_pred_lin, 'metrics', [MSE_lin, RMSE_lin, R2_lin]);

    % --- Model 2: Ridge Regression ---
    fprintf('\n--- Training Ridge Regression Model ---\n');
    lambda = 0.1; % Regularization parameter, can be tuned
    I = eye(size(X_train_lin, 2));
    I(1, 1) = 0; % Do not regularize the intercept term
    theta_ridge = pinv(X_train_lin' * X_train_lin + lambda * I) * X_train_lin' * y_train;
    y_pred_ridge = X_test_lin * theta_ridge;
    [MSE_ridge, RMSE_ridge, R2_ridge] = evaluate_model(y_test, y_pred_ridge);
    model_results{2} = struct('name', 'Ridge Regression', 'predictions', y_pred_ridge, 'metrics', [MSE_ridge, RMSE_ridge, R2_ridge]);

    % --- Model 3: K-Nearest Neighbors (KNN) Regression ---
    fprintf('\n--- Training K-Nearest Neighbors (KNN) Regression ---\n');
    % KNN requires a hyperparameter k (number of neighbors).
    k = 5;
    y_pred_knn = zeros(size(y_test));
    for i = 1:size(X_test_scaled, 1)
        % Calculate Euclidean distances from the test point to all training points.
        distances = sum((X_train_scaled - repmat(X_test_scaled(i, :), size(X_train_scaled, 1), 1)).^2, 2);

        % Find the indices of the k nearest neighbors.
        [sorted_dists, sorted_idxs] = sort(distances);
        k_nearest_neighbors_idxs = sorted_idxs(1:k);

        % Predict the value by averaging the target values of the neighbors.
        y_pred_knn(i) = mean(y_train(k_nearest_neighbors_idxs));
    end
    [MSE_knn, RMSE_knn, R2_knn] = evaluate_model(y_test, y_pred_knn);
    model_results{3} = struct('name', 'KNN Regression', 'predictions', y_pred_knn, 'metrics', [MSE_knn, RMSE_knn, R2_knn]);

    % =========================================================================
    % 4. COMPARISON OF RESULTS
    % =========================================================================

    fprintf('\n\n--- MODEL PERFORMANCE COMPARISON ---\n');
    fprintf('%-25s | %-12s | %-12s | %-12s\n', 'Model', 'MSE', 'RMSE', 'R^2');
    fprintf('-------------------------------------------------------------\n');
    for i = 1:length(model_results)
        fprintf('%-25s | %-12.4f | %-12.4f | %-12.4f\n', ...
                model_results{i}.name, ...
                model_results{i}.metrics(1), ...
                model_results{i}.metrics(2), ...
                model_results{i}.metrics(3));
    end
    fprintf('-------------------------------------------------------------\n');

    % =========================================================================
    % 5. VISUALIZATION OF PREDICTIONS VS. ACTUALS
    % =========================================================================

    fprintf('\nGenerating prediction vs. actuals plots...\n');
    figure('name', 'Model Performance Comparison');
    for i = 1:length(model_results)
        subplot(1, 3, i);
        scatter(y_test, model_results{i}.predictions, 'filled', 'b');
        title(sprintf('%s\n(Test Set)', model_results{i}.name));
        xlabel('Actual Utility Cost');
        ylabel('Predicted Utility Cost');

        % Plot the perfect prediction line (y = x).
        max_val = max(max(y_test), max(model_results{i}.predictions));
        min_val = min(min(y_test), min(model_results{i}.predictions));
        hold on;
        plot([min_val, max_val], [min_val, max_val], 'r--', 'LineWidth', 2);
        hold off;

        legend('Predictions', 'Perfect Line');
        grid on;
    end

    fprintf('Plot generation complete. Check the figure window.\n');

catch
    fprintf('\nAn error occurred. Please ensure the following:\n');
    fprintf('1. The file "heat_exchanger_network_optimization_dataset.csv" is in the same directory as this script.\n');
    fprintf('2. The column names used in the script exactly match those in the CSV file.\n');
    disp(lasterror.message);
end

