function [ye, net, valid_tx_indices, best_delay, best_offset] = FNN_Implementation(xRx, xTx, NumPreamble_TDE, InputLength, HiddenSize, LearningRate, MaxEpochs, DelayCandidates, OffsetCandidates, ScanEpochs, ScanTrainSamples, ScanValSamples)
% FNN_Implementation_Centered:  FNN ?%
%  FFE_2pscenter ?% 1.  xRx ?2 sps
% 2. ?n ? [2*n - Half, ..., 2*n + Half]
% 3. ?0 padding ?
    %% 1. ?    if nargin < 4, InputLength = 101; end % 
    if nargin < 5, HiddenSize = 64; end   
    if nargin < 6, LearningRate = 0.0005; end
    if nargin < 7, MaxEpochs = 50; end
    if nargin < 8 || isempty(DelayCandidates), DelayCandidates = -30:30; end
    if nargin < 9 || isempty(OffsetCandidates), OffsetCandidates = [1 2]; end % FFEffsetelay
    
    % ?GPU
    useGPU = canUseGPU() && (exist('gpuArray', 'file') == 2);
    if useGPU
        execEnv = 'gpu';
    else
        execEnv = 'cpu';
    end

    %% 2. data prep
    Rx_Data = xRx(:);
    Tx_Data = xTx(:);
    
    % ?
    y_mean = mean(Tx_Data);
    y_std  = std(Tx_Data);
    
    Rx_Data = (Rx_Data - mean(Rx_Data)) / std(Rx_Data);
    Tx_Data = (Tx_Data - y_mean) / y_std;
    
    % ?(Padding)
    HalfFilterLen = floor(InputLength / 2);
    Padding = HalfFilterLen;
    Rx_Data_Pad = [zeros(Padding, 1); Rx_Data; zeros(Padding, 1)];

    %% 3. ?(Linear Probe) ?    % x ?Tx ?    disp('    [FNN] Scanning best delay using Linear Probe...');
    
    best_mse = inf;
    best_delay = 0;
    best_offset = OffsetCandidates(1);
    
    % 
    ProbeLen = min(5000, NumPreamble_TDE);
    
    for oi = 1:numel(OffsetCandidates)
        offset = OffsetCandidates(oi);
        for delay = DelayCandidates
            [X_probe, Y_probe, ~] = build_centered_dataset(Rx_Data_Pad, Tx_Data, InputLength, delay, ProbeLen, Padding, offset);

            if isempty(Y_probe), continue; end

            % ?w = X \ Y
            X_probe = X_probe.'; % [Samples x Feat]
            Y_probe = Y_probe.';

            if size(X_probe, 1) > size(X_probe, 2)
                w = X_probe \ Y_probe;
                mse = mean((X_probe * w - Y_probe).^2);

                if mse < best_mse
                    best_mse = mse;
                    best_delay = delay;
                    best_offset = offset;
                end
            end
        end
    end
    
    disp(['    [FNN] Best Offset = ', num2str(best_offset), ', Best Delay = ', num2str(best_delay), ', Linear MSE = ', num2str(best_mse)]);
    disp(['[FNN] FINAL best_offset=', num2str(best_offset), ', best_delay=', num2str(best_delay)]);

    %% 4. training data
    [X_Train_Raw, Y_Train_Raw, ~] = build_centered_dataset(Rx_Data_Pad, Tx_Data, InputLength, best_delay, NumPreamble_TDE, Padding, best_offset);
    
    X_Train = X_Train_Raw.';
    Y_Train = Y_Train_Raw.';

    %% 5. ?    disp('    [FNN] Training Network...');
    
    layers = [
        featureInputLayer(InputLength, 'Normalization', 'none', 'Name', 'input')
        fullyConnectedLayer(HiddenSize, 'Name', 'fc1')
        tanhLayer('Name', 'tanh1')
        fullyConnectedLayer(32, 'Name', 'fc2')
        tanhLayer('Name', 'tanh2')
        fullyConnectedLayer(1, 'Name', 'output')
        regressionLayer('Name', 'loss')
    ];

    options = trainingOptions('adam', ...
        'MaxEpochs', MaxEpochs, ...
        'MiniBatchSize', 512, ... %  Batch Size 
        'InitialLearnRate', LearningRate, ...
        'Shuffle', 'every-epoch', ...
        'Plots', 'none', ...
        'Verbose', 0, ...
        'ExecutionEnvironment', execEnv);

    tic;
    net = trainNetwork(X_Train, Y_Train, layers, options);
    disp(['    [FNN] Training Done in ', num2str(toc), 's']);

    %% 6.  (Inference)
    disp('    [FNN] Inference on Full Sequence...');
    
    %  ()
    [X_Full, ~, valid_tx_indices] = build_centered_dataset(Rx_Data_Pad, Tx_Data, InputLength, best_delay, [], Padding, best_offset);
    
    X_Test = X_Full.';
    
    % 
    ye_pred = predict(net, X_Test, 'MiniBatchSize', 8192, 'ExecutionEnvironment', execEnv);
    ye_pred = double(ye_pred');
    
    % denormalize
    ye_val = ye_pred * y_std + y_mean;
    
    % return valid samples only
    ye = ye_val(:); % [N_valid x 1]
    valid_tx_indices = valid_tx_indices(:); % [N_valid x 1]
    
end

function [X, Y, tx_indices] = build_centered_dataset(Rx_Pad, Tx, InputLength, delay, max_samples, Padding, offset)
    %  FFE_2pscenter 
    % Rx_Pad  Padding
    %  Tx 
    
    HalfLen = floor(InputLength / 2);
    
    %  Tx ?    % ?Rx ?2 2*n
    % ?2*n - HalfLen + Padding > 0 ?...
    
    total_syms = length(Tx);
    
    % ?n
    n_candidates = (1:total_syms)';
    
    %  delay 
    %  delay Tx(n)  Rx(2*(n+delay)) 
    %  Tx 
    
    % ?Tx(n)
    % ?Rx  2*n + shift
    % shift ?delay 
    
    rx_center_indices = 2 * (n_candidates + delay) + Padding + offset;
    
    persistent printed;
    if isempty(printed)
        disp(['    [FNN] offset used = ', num2str(offset), ', delay used = ', num2str(delay)]);
        printed = true;
    end
    
    start_indices = rx_center_indices - HalfLen;
    end_indices   = rx_center_indices + HalfLen;
    
    valid_mask = (start_indices >= 1) & (end_indices <= length(Rx_Pad)) & ...
                 (n_candidates >= 1) & (n_candidates <= total_syms);
             
    valid_n = n_candidates(valid_mask);
    valid_starts = start_indices(valid_mask);
    
    % 
    if ~isempty(max_samples)
        num_keep = min(length(valid_n), max_samples);
        valid_n = valid_n(1:num_keep);
        valid_starts = valid_starts(1:num_keep);
    end
    
    num_valid = length(valid_n);
    X = zeros(InputLength, num_valid, 'single');
    
    %  X (?
    % FE ?(idx_start : -1 : idx_end) 
    % FNN 
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



