function [ye, net, valid_tx_indices, best_delay, best_offset] = RNN_Advanced_Implementation( ...
    xRx, xTx, NumPreamble_TDE, InputLength, HiddenSize, LearningRate, MaxEpochs, k, ...
    DelayCandidates, OffsetCandidates)
% RNN_Advanced_Implementation (Residual AR-RNN)
%
% 核心架构: Output = Linear_FFE_DFE(Input) + NN_Residual(Input)
% 这种 "Residual" 结构能确保性能 >= 线性均衡器，并大幅降低 NN 的训练难度。

    %% ===== defaults =====
    if nargin < 4 || isempty(InputLength), InputLength = 101; end
    if nargin < 5 || isempty(HiddenSize), HiddenSize = 48; end % 适当减小，专注拟合残差
    if nargin < 6 || isempty(LearningRate), LearningRate = 1e-3; end
    if nargin < 7 || isempty(MaxEpochs), MaxEpochs = 40; end
    if nargin < 8 || isempty(k), k = 15; end % 增加反馈深度 (Windowed Decision)
    if nargin < 9 || isempty(DelayCandidates), DelayCandidates = -20:20; end
    if nargin < 10 || isempty(OffsetCandidates), OffsetCandidates = [1 2]; end

    execEnv = 'auto';
    try
        if canUseGPU() && (exist('gpuArray', 'file') == 2), execEnv = 'gpu'; end
    catch, execEnv = 'cpu'; end

    %% ===== Preprocess =====
    Rx = xRx(:);
    Tx = xTx(:);
    
    % Z-score Norm
    y_mean = mean(Tx);
    y_std  = std(Tx);
    Rx = (Rx - mean(Rx)) / std(Rx);
    Tx_n = (Tx - y_mean) / y_std;

    %% ===== 1) Linear Probe & Alignment (Using RLS/Ridge) =====
    % 这一步非常关键：先找到最优的线性 FFE/DFE 权重和时延
    % disp('    [Adv-RNN] Step 1: Solving Linear FFE/DFE Baseline...');
    
    best_mse = inf;
    best_delay = 0;
    best_offset = OffsetCandidates(1);
    w_lin_best = [];
    
    % 使用部分数据进行快速扫描
    ScanSamples = min(4000, NumPreamble_TDE);
    
    for offset = OffsetCandidates
        for delay = DelayCandidates
            % 构建 AR 数据矩阵: [Rx_Window, FB_Window]
            % 注意：探测阶段 FB 使用真实的 Tx (Teacher Forcing)
            [X_ar, Y_ar] = build_ar_dataset(Rx, Tx_n, InputLength, k, offset, delay, ScanSamples);
            
            if size(X_ar, 2) < 500, continue; end
            
            X_t = X_ar.';
            Y_t = Y_ar.';
            
            % Ridge Regression (闭式解)
            % w = (X'X + lam*I)^-1 X'Y
            R = X_t' * X_t;
            w = (R + 1e-3*eye(size(R))) \ (X_t' * Y_t);
            
            mse = mean((X_t * w - Y_t).^2);
            
            if mse < best_mse
                best_mse = mse;
                best_delay = delay;
                best_offset = offset;
                w_lin_best = w;
            end
        end
    end
    % disp(['    [Adv-RNN] Linear Baseline MSE: ', num2str(best_mse), ' (Delay=', num2str(best_delay), ')']);

    %% ===== 2) Prepare Training Data for NN (Residuals) =====
    % 使用全量 Preamble
    [X_all, Y_all, valid_tx_indices] = build_ar_dataset(Rx, Tx_n, InputLength, k, best_offset, best_delay, NumPreamble_TDE);
    
    X_train = X_all.';
    Y_train = Y_all.';
    
    % 计算线性部分的预测值
    Y_lin_pred = X_train * w_lin_best;
    
    % === 关键: 计算残差 ===
    Residual_Target = Y_train - Y_lin_pred;
    % ======================
    
    %% ===== 3) Train NN on Residuals =====
    % disp('    [Adv-RNN] Step 2: Training NN on Residuals...');
    
    layers = [
        featureInputLayer(InputLength + k, 'Normalization', 'none', 'Name', 'input')
        
        fullyConnectedLayer(HiddenSize, 'Name', 'fc1', 'WeightsInitializer', 'he')
        tanhLayer('Name', 'act1')
        
        fullyConnectedLayer(ceil(HiddenSize/2), 'Name', 'fc2', 'WeightsInitializer', 'he')
        tanhLayer('Name', 'act2')
        
        fullyConnectedLayer(1, 'Name', 'output')
        regressionLayer('Name', 'loss')
    ];

    options = trainingOptions('adam', ...
        'MaxEpochs', MaxEpochs, ... 
        'MiniBatchSize', 512, ... 
        'InitialLearnRate', LearningRate, ...
        'L2Regularization', 1e-4, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', 0, ...
        'Plots', 'none', ...
        'ExecutionEnvironment', execEnv);
    
    net = trainNetwork(X_train, Residual_Target, layers, options);
    
    %% ===== 4) Final Inference (Decision Feedback Loop) =====
    % disp('    [Adv-RNN] Step 3: Inference with Decision Feedback...');
    
    % 准备推断
    % 我们需要手动进行循环，因为每一步的 Feedback 依赖于上一步的总输出（线性+非线性）
    
    % 1. 预先构建 Rx 部分的矩阵 (Rx 不依赖反馈，是固定的)
    [Rx_Mat, ~, all_valid_indices] = build_rx_matrix_only(Rx, Tx_n, InputLength, best_offset, best_delay);
    % Rx_Mat: [InputLength x N_total]
    
    N_total = size(Rx_Mat, 2);
    ye_total = zeros(N_total, 1);
    
    % 学习到的 PAM4 电平 (用于硬判决)
    % 从 Y_train 统计
    [~, centers] = kmeans(double(Y_train(1:min(2000,end))), 4, 'Replicates', 3);
    pam_levels = sort(centers).';
    thr = (pam_levels(1:3) + pam_levels(2:4))/2;
    
    % 提取 NN 权重用于手动前传 (比 predict 快且方便单步循环)
    W1 = gather(net.Layers(2).Weights); b1 = gather(net.Layers(2).Bias);
    W2 = gather(net.Layers(4).Weights); b2 = gather(net.Layers(4).Bias);
    W3 = gather(net.Layers(6).Weights); b3 = gather(net.Layers(6).Bias);
    
    % Feedback Buffer (Initialize with zeros or known preamble if available)
    % 这里简单起见用 0 初始化，或者用训练集最后几个数据
    fb_buffer = zeros(k, 1);
    if k > 0 && ~isempty(Y_train)
        fb_buffer = Y_train(end-k+1:end); 
    end
    
    % 循环处理
    for n = 1 : N_total
        % 当前 Rx 窗口
        rx_vec = Rx_Mat(:, n); % [InputLength x 1]
        
        % 构建完整输入向量: [Rx; FB]
        input_vec = [rx_vec; fb_buffer]; % [(InputLength+k) x 1]
        
        % 1. 线性预测
        y_lin = w_lin_best.' * input_vec;
        
        % 2. NN 残差预测 (Manual Forward)
        h1 = tanh(W1 * input_vec + b1);
        h2 = tanh(W2 * h1 + b2);
        y_res = W3 * h2 + b3;
        
        % 3. 总输出
        y_out = y_lin + y_res;
        ye_total(n) = y_out;
        
        % 4. 硬判决 (用于更新 Feedback)
        y_hard = hard_dec(y_out, pam_levels, thr);
        
        % 5. 更新 Buffer (Shift & Push)
        fb_buffer = [y_hard; fb_buffer(1:end-1)];
    end
    
    % 反归一化
    ye_final = ye_total * y_std + y_mean;
    
    % 映射回全长数组
    ye = zeros(length(Tx), 1);
    ye(all_valid_indices) = ye_final;
    
end

%% ===== Helpers =====

function [X, Y, valid_idx] = build_ar_dataset(Rx, Tx, InputLen, k, offset, delay, max_samples)
    % 构建用于训练的 AR 数据集 (Teacher Forcing: Feedback 来自真实 Tx)
    
    HalfLen = floor(InputLen/2);
    TotalSyms = length(Tx);
    
    % 确定有效范围
    % 我们需要 Rx[n-Half..n+Half] 和 Tx[n-1..n-k]
    % 所以 n 必须 >= k + 1
    
    start_n = k + 1;
    end_n = TotalSyms;
    
    if ~isempty(max_samples)
        end_n = min(end_n, start_n + max_samples - 1);
    end
    
    n_indices = (start_n : end_n).';
    
    % Rx Center Indices
    rx_centers = 2 * (n_indices + delay) + offset;
    
    % 检查边界
    rx_starts = rx_centers - HalfLen;
    rx_ends   = rx_centers + HalfLen;
    
    valid_mask = (rx_starts >= 1) & (rx_ends <= length(Rx));
    valid_n = n_indices(valid_mask);
    valid_starts = rx_starts(valid_mask);
    
    NumValid = length(valid_n);
    X = zeros(InputLen + k, NumValid, 'single');
    Y = single(Tx(valid_n)).'; % Transpose to [1 x N]
    
    for i = 1 : NumValid
        n_curr = valid_n(i);
        s_idx = valid_starts(i);
        
        % Rx Part
        rx_part = Rx(s_idx : s_idx + InputLen - 1);
        
        % Feedback Part (True Tx)
        fb_part = Tx(n_curr-1 : -1 : n_curr-k);
        
        X(:, i) = [rx_part; fb_part];
    end
    
    valid_idx = valid_n;
end

function [RxMat, TxMat, valid_idx] = build_rx_matrix_only(Rx, Tx, InputLen, offset, delay)
    % 仅构建 Rx 部分矩阵，用于 Inference 阶段
    HalfLen = floor(InputLen/2);
    TotalSyms = length(Tx);
    
    n_indices = (1 : TotalSyms).';
    rx_centers = 2 * (n_indices + delay) + offset;
    
    rx_starts = rx_centers - HalfLen;
    rx_ends   = rx_centers + HalfLen;
    
    valid_mask = (rx_starts >= 1) & (rx_ends <= length(Rx));
    valid_n = n_indices(valid_mask);
    valid_starts = rx_starts(valid_mask);
    
    NumValid = length(valid_n);
    RxMat = zeros(InputLen, NumValid, 'single');
    
    for i = 1 : NumValid
        s_idx = valid_starts(i);
        RxMat(:, i) = Rx(s_idx : s_idx + InputLen - 1);
    end
    
    TxMat = Tx(valid_n); % Not used usually
    valid_idx = valid_n;
end

function yq = hard_dec(y, levels, thr)
    % 标量硬判决
    if y < thr(1), yq = levels(1);
    elseif y < thr(2), yq = levels(2);
    elseif y < thr(3), yq = levels(3);
    else, yq = levels(4);
    end
end

function flag = canUseGPU()
    try, d = gpuDevice; flag = d.Index > 0; catch, flag = false; end
end
