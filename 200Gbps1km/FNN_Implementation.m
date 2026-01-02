function [ye, net, valid_tx_indices, best_delay, best_offset] = FNN_Implementation(xRx, xTx, NumPreamble_TDE, InputLength, HiddenSize, LearningRate, MaxEpochs, DelayCandidates, OffsetCandidates, init_net)
    %% 0. Parameters & Defaults
    if nargin < 4, InputLength = 111; end 
    if nargin < 5, HiddenSize = 32; end    
    if nargin < 6, LearningRate = 0.001; end
    if nargin < 7, MaxEpochs = 50; end
    if nargin < 8 || isempty(DelayCandidates), DelayCandidates = -30:30; end 
    if nargin < 9 || isempty(OffsetCandidates), OffsetCandidates = [1 2]; end
    if nargin < 10, init_net = []; end
    
    execEnv = 'auto'; 
    try
        if canUseGPU() && (exist('gpuArray', 'file') == 2)
             execEnv = 'gpu';
        end
    catch
        execEnv = 'cpu';
    end

    %% 1. Data Preprocessing
    Rx_Data = xRx(:);
    Tx_Data = xTx(:);
    
    % Z-score Normalization
    y_mean = mean(Tx_Data);
    y_std  = std(Tx_Data);
    Rx_mean = mean(Rx_Data);
    Rx_std = std(Rx_Data);
    
    Rx_Data = (Rx_Data - Rx_mean) / Rx_std;
    Tx_Data = (Tx_Data - y_mean) / y_std;
    
    HalfFilterLen = floor(InputLength / 2);
    Padding = HalfFilterLen;
    Rx_Data_Pad = [zeros(Padding, 1); Rx_Data; zeros(Padding, 1)];

    %% 2. Fast Linear Probe (Using ALL Preamble Data)
    ProbeLen = NumPreamble_TDE; 
    
    best_mse = inf;
    best_delay = 0;
    best_offset = OffsetCandidates(1);
    
    % Use RLS-like pre-calculation or just simple Ridge
    for oi = 1:numel(OffsetCandidates)
        offset = OffsetCandidates(oi);
        for delay = DelayCandidates
            [X_probe, Y_probe, ~] = build_centered_dataset(Rx_Data_Pad, Tx_Data, InputLength, delay, ProbeLen, Padding, offset);
            if isempty(Y_probe), continue; end
            X_probe = X_probe.'; 
            Y_probe = Y_probe.';
            
            % Ridge Regression
            if size(X_probe, 1) > size(X_probe, 2)
                % Regularization slightly higher to be safe
                w = (X_probe' * X_probe + 1e-2*eye(size(X_probe,2))) \ (X_probe' * Y_probe);
                mse = mean((X_probe * w - Y_probe).^2);
                if mse < best_mse
                    best_mse = mse;
                    best_delay = delay;
                    best_offset = offset;
                end
            end
        end
    end
    
    % disp(['    [FNN Probe] Best MSE: ', num2str(best_mse), ' Delay: ', num2str(best_delay)]);

    %% 3. Build Training Set
    [X_All_Preamble, Y_All_Preamble, ~] = build_centered_dataset(Rx_Data_Pad, Tx_Data, InputLength, best_delay, NumPreamble_TDE, Padding, best_offset);
    
    X_Total = X_All_Preamble.';
    Y_Total = Y_All_Preamble.';
    
    % Shuffle
    num_total = size(X_Total, 1);
    perm = randperm(num_total);
    X_Total = X_Total(perm, :);
    Y_Total = Y_Total(perm, :);

    % Train/Val Split
    num_train = floor(0.9 * num_total);
    X_Train = X_Total(1:num_train, :);
    Y_Train = Y_Total(1:num_train, :);
    X_Val   = X_Total(num_train+1:end, :);
    Y_Val   = Y_Total(num_train+1:end, :);

    %% 4. Network Architecture
    if isempty(init_net)
        layers = [
            featureInputLayer(InputLength, 'Normalization', 'none', 'Name', 'input')
            
            % Linear FNN (Deep Linear Network)
            % This has been verified to match FFE performance on this dataset.
            % Non-linear activations (ReLU/Tanh) degraded performance due to 
            % the linear nature of the channel and optimization difficulties.
            fullyConnectedLayer(HiddenSize, 'Name', 'fc1', 'WeightsInitializer', 'he') 
            % leakyReluLayer(0.1, 'Name', 'act1') 
            
            fullyConnectedLayer(1, 'Name', 'output')
            regressionLayer('Name', 'loss')
        ];
        initial_network = layers;
    else
        initial_network = init_net.Layers;
    end

    options = trainingOptions('adam', ...
        'MaxEpochs', MaxEpochs, ... 
        'MiniBatchSize', 256, ... 
        'InitialLearnRate', LearningRate, ...
        'L2Regularization', 1e-4, ... 
        'ValidationData', {X_Val, Y_Val}, ...
        'ValidationPatience', 10, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', 0, ...
        'Plots', 'none', ...
        'ExecutionEnvironment', execEnv);
    
    if isempty(init_net)
         [net, ~] = trainNetwork(X_Train, Y_Train, initial_network, options);
    else
         [net, ~] = trainNetwork(X_Train, Y_Train, init_net.Layers, options);
    end

    %% 5. Inference
    [X_Full, ~, valid_tx_indices] = build_centered_dataset(Rx_Data_Pad, Tx_Data, InputLength, best_delay, [], Padding, best_offset);
    X_Test = X_Full.';
    
    ye_pred = predict(net, X_Test, 'MiniBatchSize', 8192, 'ExecutionEnvironment', execEnv);
    ye_pred = double(ye_pred');
    
    ye = ye_pred * y_std + y_mean;
    ye = ye(:); 
    valid_tx_indices = valid_tx_indices(:); 
end

function [X, Y, tx_indices] = build_centered_dataset(Rx_Pad, Tx, InputLength, delay, max_samples, Padding, offset)
    HalfLen = floor(InputLength / 2);
    total_syms = length(Tx);
    n_candidates = (1:total_syms)';
    
    rx_center_indices = 2 * (n_candidates + delay) + Padding + offset;
    
    start_indices = rx_center_indices - HalfLen;
    end_indices   = rx_center_indices + HalfLen;
    
    valid_mask = (start_indices >= 1) & (end_indices <= length(Rx_Pad)) & ...
                 (n_candidates >= 1) & (n_candidates <= total_syms);
             
    valid_n = n_candidates(valid_mask);
    valid_starts = start_indices(valid_mask);
    
    if ~isempty(max_samples)
        num_keep = min(length(valid_n), max_samples);
        valid_n = valid_n(1:num_keep);
        valid_starts = valid_starts(1:num_keep);
    end
    
    num_valid = length(valid_n);
    X = zeros(InputLength, num_valid, 'single');
    
    for i = 1:num_valid
        s_idx = valid_starts(i);
        X(:, i) = Rx_Pad(s_idx : s_idx + InputLength - 1);
    end
    Y = single(Tx(valid_n)).';
    tx_indices = valid_n;
end

function flag = canUseGPU()
    try
        d = gpuDevice;
        flag = d.Index > 0;
    catch
        flag = false;
    end
end