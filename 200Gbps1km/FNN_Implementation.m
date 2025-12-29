function [ye, net, valid_tx_indices] = FNN_Implementation(xRx, xTx, NumPreamble_TDE, InputLength, HiddenSize, LearningRate, MaxEpochs, DelayCandidates, OffsetCandidates, ScanEpochs, ScanTrainSamples, ScanValSamples)
% FNN_Implementation_Centered: 中心窗结构的 FNN 均衡器
%
% 模仿 FFE_2pscenter 的数据构造逻辑：
% 1. 输入 xRx 为 2 sps
% 2. 窗口以当前符号 n 为中心: [2*n - Half, ..., 2*n + Half]
% 3. 严格对齐输出索引，避免 0 padding 污染归一化

    %% 1. 参数默认值
    if nargin < 4, InputLength = 101; end % 建议奇数
    if nargin < 5, HiddenSize = 64; end   
    if nargin < 6, LearningRate = 0.0005; end
    if nargin < 7, MaxEpochs = 50; end
    if nargin < 8 || isempty(DelayCandidates), DelayCandidates = -30:30; end
    if nargin < 9 || isempty(OffsetCandidates), OffsetCandidates = 0; end % FFE逻辑通常不需要额外offset，依靠delay
    
    % 检查 GPU
    useGPU = canUseGPU() && (exist('gpuArray', 'file') == 2);
    if useGPU
        execEnv = 'gpu';
    else
        execEnv = 'cpu';
    end

    %% 2. 数据预处理
    Rx_Data = xRx(:);
    Tx_Data = xTx(:);
    
    % 归一化输入/输出
    y_mean = mean(Tx_Data);
    y_std  = std(Tx_Data);
    
    Rx_Data = (Rx_Data - mean(Rx_Data)) / std(Rx_Data);
    Tx_Data = (Tx_Data - y_mean) / y_std;
    
    % 补零以处理边界 (Padding)
    HalfFilterLen = floor(InputLength / 2);
    Padding = HalfFilterLen;
    Rx_Data_Pad = [zeros(Padding, 1); Rx_Data; zeros(Padding, 1)];

    %% 3. 快速线性探测 (Linear Probe) 寻找最佳时延
    % 即使是中心窗，Rx 和 Tx 之间可能仍有群时延差异
    disp('    [FNN] Scanning best delay using Linear Probe...');
    
    best_mse = inf;
    best_delay = 0;
    
    % 使用较小的训练集进行探测
    ProbeLen = min(5000, NumPreamble_TDE);
    
    for delay = DelayCandidates
        [X_probe, Y_probe, ~] = build_centered_dataset(Rx_Data_Pad, Tx_Data, InputLength, delay, ProbeLen, Padding);
        
        if isempty(Y_probe), continue; end
        
        % 求解线性权重 w = X \ Y
        X_probe = X_probe.'; % [Samples x Feat]
        Y_probe = Y_probe.';
        
        if size(X_probe, 1) > size(X_probe, 2)
            w = X_probe \ Y_probe;
            mse = mean((X_probe * w - Y_probe).^2);
            
            if mse < best_mse
                best_mse = mse;
                best_delay = delay;
            end
        end
    end
    
    disp(['    [FNN] Best Delay = ', num2str(best_delay), ', Linear MSE = ', num2str(best_mse)]);

    %% 4. 构建训练数据集
    [X_Train_Raw, Y_Train_Raw, ~] = build_centered_dataset(Rx_Data_Pad, Tx_Data, InputLength, best_delay, NumPreamble_TDE, Padding);
    
    X_Train = X_Train_Raw.';
    Y_Train = Y_Train_Raw.';

    %% 5. 网络构建与训练
    disp('    [FNN] Training Network...');
    
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
        'MiniBatchSize', 512, ... % 调小 Batch Size 提高泛化
        'InitialLearnRate', LearningRate, ...
        'Shuffle', 'every-epoch', ...
        'Plots', 'none', ...
        'Verbose', 0, ...
        'ExecutionEnvironment', execEnv);

    tic;
    net = trainNetwork(X_Train, Y_Train, layers, options);
    disp(['    [FNN] Training Done in ', num2str(toc), 's']);

    %% 6. 全量推理 (Inference)
    disp('    [FNN] Inference on Full Sequence...');
    
    % 构建全量输入 (不限制样本数)
    [X_Full, ~, valid_tx_indices] = build_centered_dataset(Rx_Data_Pad, Tx_Data, InputLength, best_delay, [], Padding);
    
    X_Test = X_Full.';
    
    % 预测
    ye_pred = predict(net, X_Test, 'MiniBatchSize', 8192, 'ExecutionEnvironment', execEnv);
    ye_pred = double(ye_pred');
    
    % 反归一化
    ye_val = ye_pred * y_std + y_mean;
    
    % 构造输出
    % 这里我们不再填充 0，而是直接返回 valid 部分的值和索引
    % 主程序负责利用 valid_tx_indices 进行对齐比较
    ye = ye_val(:); % [N_valid x 1]
    valid_tx_indices = valid_tx_indices(:); % [N_valid x 1]
    
end

function [X, Y, tx_indices] = build_centered_dataset(Rx_Pad, Tx, InputLength, delay, max_samples, Padding)
    % 模仿 FFE_2pscenter 的取数逻辑
    % Rx_Pad 已经两头补了 Padding
    % 目标 Tx 索引范围
    
    HalfLen = floor(InputLength / 2);
    
    % 原始 Tx 数据的有效索引范围
    % 考虑到 Rx 是 2倍采样，中心点是 2*n
    % 实际上我们需要 2*n - HalfLen + Padding > 0 且 ...
    
    total_syms = length(Tx);
    
    % 生成候选符号索引 n
    n_candidates = (1:total_syms)';
    
    % 加上 delay 修正
    % 这里 delay 定义为：Tx(n) 对应 Rx(2*(n+delay)) 附近的窗
    % 或者更简单的：我们对 Tx 索引进行平移
    
    % 让我们定义：我们要预测 Tx(n)
    % 对应的 Rx 中心点是 2*n + shift
    % shift 由 delay 参数控制
    
    rx_center_indices = 2 * (n_candidates + delay) + Padding;
    
    % 检查边界
    start_indices = rx_center_indices - HalfLen;
    end_indices   = rx_center_indices + HalfLen;
    
    valid_mask = (start_indices >= 1) & (end_indices <= length(Rx_Pad)) & ...
                 (n_candidates >= 1) & (n_candidates <= total_syms);
             
    valid_n = n_candidates(valid_mask);
    valid_starts = start_indices(valid_mask);
    
    % 截断数量
    if ~isempty(max_samples)
        num_keep = min(length(valid_n), max_samples);
        valid_n = valid_n(1:num_keep);
        valid_starts = valid_starts(1:num_keep);
    end
    
    num_valid = length(valid_n);
    X = zeros(InputLength, num_valid, 'single');
    
    % 填充 X (列优先)
    % 注意：FFE 代码里是倒序取 (idx_start : -1 : idx_end) 模拟卷积
    % FNN 只要顺序一致即可，这里我们用正序，网络会自动适应
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