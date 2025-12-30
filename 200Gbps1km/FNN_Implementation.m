function [ye, net, valid_tx_indices, best_delay, best_offset] = FNN_Implementation(xRx, xTx, NumPreamble_TDE, InputLength, HiddenSize, LearningRate, MaxEpochs, DelayCandidates, OffsetCandidates)
    %% 0. 参数鲁棒性检查与高观点调整
    % 建议：如果外部传入101，这里强制内部截断或者后续增加强正则化。
    % 对于FNN，输入长度过长是过拟合核心原因。
    if nargin < 4, InputLength = 51; end % 主动建议减小 InputLength
    if nargin < 5, HiddenSize = 64; end   
    if nargin < 6, LearningRate = 0.001; end
    if nargin < 7, MaxEpochs = 50; end
    if nargin < 8 || isempty(DelayCandidates), DelayCandidates = -30:30; end
    if nargin < 9 || isempty(OffsetCandidates), OffsetCandidates = [1 2]; end
    
    useGPU = canUseGPU() && (exist('gpuArray', 'file') == 2);
    execEnv = 'auto'; % 让MATLAB自己决定，通常更稳健

    %% 1. 数据预处理 (Data Prep)
    Rx_Data = xRx(:);
    Tx_Data = xTx(:);
    
    % 标准化 (Z-score normalization)
    % 注意：光通信信号往往非高斯，但Z-score通常足够
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

    %% 2. 快速线性探测 (Linear Probe for Delay/Offset)
    % 这一步非常棒，保持原样，它是寻找最佳相位的关键
    disp('    [FNN] Scanning best delay using Linear Probe...');
    best_mse = inf;
    best_delay = 0;
    best_offset = OffsetCandidates(1);
    ProbeLen = min(4000, NumPreamble_TDE); % 用少一点数据定相位即可
    
    for oi = 1:numel(OffsetCandidates)
        offset = OffsetCandidates(oi);
        for delay = DelayCandidates
            [X_probe, Y_probe, ~] = build_centered_dataset(Rx_Data_Pad, Tx_Data, InputLength, delay, ProbeLen, Padding, offset);
            if isempty(Y_probe), continue; end
            X_probe = X_probe.'; 
            Y_probe = Y_probe.';
            % 简单的最小二乘解
            if size(X_probe, 1) > size(X_probe, 2)
                % 加个微小正则化防止奇异
                w = (X_probe' * X_probe + 1e-4*eye(size(X_probe,2))) \ (X_probe' * Y_probe);
                mse = mean((X_probe * w - Y_probe).^2);
                if mse < best_mse
                    best_mse = mse;
                    best_delay = delay;
                    best_offset = offset;
                end
            end
        end
    end
    disp(['    [FNN] Optimal: Offset=', num2str(best_offset), ', Delay=', num2str(best_delay)]);

    %% 3. 构建训练集与验证集 (Train/Val Split) - 关键修改！
    % 使用所有 Preamble 数据
    [X_All_Preamble, Y_All_Preamble, ~] = build_centered_dataset(Rx_Data_Pad, Tx_Data, InputLength, best_delay, NumPreamble_TDE, Padding, best_offset);
    
    % 转换为 double 确保精度（虽然 single 更快，但在 MATLAB 混合精度下有时会有问题）
    X_Total = X_All_Preamble.';
    Y_Total = Y_All_Preamble.';
    
    % 随机切分 80% 训练，20% 验证
    num_total = size(X_Total, 1);
    cv = cvpartition(num_total, 'HoldOut', 0.2);
    idxTrain = cv.training;
    idxVal = cv.test;
    
    X_Train = X_Total(idxTrain, :);
    Y_Train = Y_Total(idxTrain, :);
    X_Val   = X_Total(idxVal, :);
    Y_Val   = Y_Total(idxVal, :);

    %% 4. 网络构建与训练 (Network & Training)
    disp('    [FNN] Training with Regularization & Early Stopping...');
    
    layers = [
        featureInputLayer(InputLength, 'Normalization', 'none', 'Name', 'input')
        
        % 第一层：用较少的神经元，起到类似 PCA 降维的作用，强迫提取关键特征
        fullyConnectedLayer(64, 'Name', 'fc1', 'WeightsInitializer', 'he') 
        tanhLayer('Name', 'act1')
        dropoutLayer(0.2, 'Name', 'drop1') % 提高 Dropout 到 20%
        
        % 第二层
        fullyConnectedLayer(32, 'Name', 'fc2', 'WeightsInitializer', 'he')
        tanhLayer('Name', 'act2')
        % dropoutLayer(0.1, 'Name', 'drop2') % 可选
        
        fullyConnectedLayer(1, 'Name', 'output')
        regressionLayer('Name', 'loss')
    ];

    options = trainingOptions('adam', ...
        'MaxEpochs', 60, ... % 稍微多给点时间
        'MiniBatchSize', 512, ... 
        'InitialLearnRate', 0.001, ...
        'L2Regularization', 1e-3, ... % ！！！关键：增加 L2 正则化力度 (从 1e-4 -> 1e-3)
        'ValidationData', {X_Val, Y_Val}, ...
        'ValidationPatience', 8, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', 0, ...
        'Plots', 'none', ...
        'ExecutionEnvironment', execEnv);
    
    tic;
    [net, info] = trainNetwork(X_Train, Y_Train, layers, options);
    trainTime = toc;
    
    % 打印最终的验证集 RMSE，作为调试参考
    valRMSE = info.ValidationRMSE(end);
    if isnan(valRMSE), valRMSE = info.TrainingRMSE(end); end
    disp(['    [FNN] Done. Time: ', num2str(trainTime, '%.2f'), 's. Final Val RMSE: ', num2str(valRMSE)]);

    %% 5. 全局推断 (Inference)
    disp('    [FNN] Inference on Full Sequence...');
    
    % 获取全部数据用于测试
    [X_Full, ~, valid_tx_indices] = build_centered_dataset(Rx_Data_Pad, Tx_Data, InputLength, best_delay, [], Padding, best_offset);
    X_Test = X_Full.';
    
    % 预测
    ye_pred = predict(net, X_Test, 'MiniBatchSize', 8192, 'ExecutionEnvironment', execEnv);
    ye_pred = double(ye_pred');
    
    % 反归一化
    ye_val = ye_pred * y_std + y_mean;
    
    ye = ye_val(:); 
    valid_tx_indices = valid_tx_indices(:); 
end

%% 辅助函数 (保持不变)
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