function [ye_combined, net, valid_tx_indices, best_delay, best_offset] = FNN_Residual_Implementation(xRx, xTx, NumPreamble_TDE, InputLength, HiddenSize, LearningRate, MaxEpochs, DelayCandidates, OffsetCandidates)
    %% 0. 参数初始化
    if nargin < 4, InputLength = 101; end
    if nargin < 5, HiddenSize = 64; end
    if nargin < 6, LearningRate = 0.001; end
    if nargin < 7, MaxEpochs = 50; end
    if nargin < 8 || isempty(DelayCandidates), DelayCandidates = -30:30; end
    if nargin < 9 || isempty(OffsetCandidates), OffsetCandidates = [1 2]; end
    
    useGPU = canUseGPU() && (exist('gpuArray', 'file') == 2);
    execEnv = 'auto';

    %% 1. 数据预处理
    Rx_Data = xRx(:);
    Tx_Data = xTx(:);
    
    % 简单的 Z-score 标准化
    y_mean = mean(Tx_Data);
    y_std  = std(Tx_Data);
    Rx_mean = mean(Rx_Data);
    Rx_std = std(Rx_Data);
    
    Rx_Data = (Rx_Data - Rx_mean) / Rx_std;
    Tx_Data = (Tx_Data - y_mean) / y_std;
    
    % Padding
    HalfFilterLen = floor(InputLength / 2);
    Padding = HalfFilterLen;
    Rx_Data_Pad = [zeros(Padding, 1); Rx_Data; zeros(Padding, 1)];

    %% 2. 线性 FFE 预训练 (Linear FFE Pre-stage)
    % 这一步非常关键：先用线性滤波器把主要的 ISI 消除
    disp('    [Res-FNN] Step 1: Solving Linear FFE...');
    
    % 同样需要扫描最佳时延，直接复用线性探测逻辑
    best_mse = inf;
    best_delay = 0;
    best_offset = OffsetCandidates(1);
    w_best = [];
    ProbeLen = min(NumPreamble_TDE, 10000); 

    for oi = 1:numel(OffsetCandidates)
        offset = OffsetCandidates(oi);
        for delay = DelayCandidates
            [X_probe, Y_probe, ~] = build_centered_dataset(Rx_Data_Pad, Tx_Data, InputLength, delay, ProbeLen, Padding, offset);
            if isempty(Y_probe), continue; end
            X_probe = X_probe.'; 
            Y_probe = Y_probe.';
            
            % 求解线性 FFE 权重 w
            % X * w = Y -> w = (X'X)\X'Y
            % 添加微小正则化
            R = (X_probe' * X_probe);
            w = (R + 1e-4 * eye(size(R))) \ (X_probe' * Y_probe);
            
            y_lin_est = X_probe * w;
            mse = mean((y_lin_est - Y_probe).^2);
            
            if mse < best_mse
                best_mse = mse;
                best_delay = delay;
                best_offset = offset;
                w_best = w;
            end
        end
    end
    disp(['    [Res-FNN] Linear Baseline MSE: ', num2str(best_mse), ' (Delay=', num2str(best_delay), ')']);

    %% 3. 准备残差训练数据 (Residual Training Data)
    % 使用所有 Preamble 数据
    [X_All, Y_All, ~] = build_centered_dataset(Rx_Data_Pad, Tx_Data, InputLength, best_delay, NumPreamble_TDE, Padding, best_offset);
    X_Total = X_All.';
    Y_Total = Y_All.';
    
    % 计算线性部分的输出
    Y_Lin_Total = X_Total * w_best;
    
    % === 核心：FNN 的目标是残差 ===
    Residual_Target = Y_Total - Y_Lin_Total;
    % ==============================

    % 划分训练/验证集
    num_total = size(X_Total, 1);
    cv = cvpartition(num_total, 'HoldOut', 0.2);
    idxTrain = cv.training;
    idxVal = cv.test;
    
    X_Train = X_Total(idxTrain, :);
    E_Train = Residual_Target(idxTrain, :); % Target is Error
    X_Val   = X_Total(idxVal, :);
    E_Val   = Residual_Target(idxVal, :);
    
    %% 4. 训练 FNN 拟合残差
    disp('    [Res-FNN] Step 2: Training FNN on Residuals...');
    
    layers = [
        featureInputLayer(InputLength, 'Normalization', 'none', 'Name', 'input')
        
        % 既然只拟合残差（非线性部分），隐藏层可以稍微小一点，或者保持原样
        fullyConnectedLayer(HiddenSize, 'Name', 'fc1', 'WeightsInitializer', 'he') 
        tanhLayer('Name', 'act1')
        % dropoutLayer(0.2, 'Name', 'drop1')
        
        fullyConnectedLayer(32, 'Name', 'fc2', 'WeightsInitializer', 'he')
        tanhLayer('Name', 'act2')
        
        fullyConnectedLayer(1, 'Name', 'output') % 输出预测的残差
        regressionLayer('Name', 'loss')
    ];

    options = trainingOptions('adam', ...
        'MaxEpochs', MaxEpochs, ... 
        'MiniBatchSize', 512, ... 
        'InitialLearnRate', LearningRate, ...
        'L2Regularization', 1e-3, ...
        'ValidationData', {X_Val, E_Val}, ...
        'ValidationPatience', 8, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', 0, ...
        'Plots', 'none', ...
        'ExecutionEnvironment', execEnv);
    
    [net, info] = trainNetwork(X_Train, E_Train, layers, options);
    disp(['    [Res-FNN] Done. Final Residual RMSE: ', num2str(info.ValidationRMSE(end))]);

    %% 5. 全局推断与合并 (Inference & Combine)
    disp('    [Res-FNN] Final Inference...');
    
    [X_Full, ~, valid_tx_indices] = build_centered_dataset(Rx_Data_Pad, Tx_Data, InputLength, best_delay, [], Padding, best_offset);
    X_Test = X_Full.';
    
    % 1. 线性部分
    y_lin_full = X_Test * w_best;
    
    % 2. 非线性残差部分
    y_res_pred = predict(net, X_Test, 'MiniBatchSize', 8192, 'ExecutionEnvironment', execEnv);
    y_res_pred = double(y_res_pred');
    y_lin_full = double(y_lin_full');
    
    % 3. 合并
    ye_pred = y_lin_full + y_res_pred; % Sum up
    
    % 反归一化
    ye_combined = ye_pred * y_std + y_mean;
    ye_combined = ye_combined(:);
    valid_tx_indices = valid_tx_indices(:);
end

%% 辅助函数 (复用)
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