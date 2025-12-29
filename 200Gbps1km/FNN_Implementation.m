function [ye, net] = FNN_Implementation(xRx, xTx, NumPreamble_TDE, InputLength, HiddenSize, LearningRate, MaxEpochs, DelayCandidates, OffsetCandidates, ScanEpochs, ScanTrainSamples, ScanValSamples)
% FNN_Implementation: 前馈神经网络均衡器 (Feedforward Neural Network Equalizer)
%
% 原理: 使用多层感知机 (MLP) 拟合逆信道特性。
% 优势: 强大的非线性拟合能力，无需预设非线性项的形式。
% 加速: 自动检测并使用 GPU (CUDA) 加速。
%
% 输入:
%   xRx:             接收信号 (2 sps, 或者是降采样后的 1 sps，取决于调用时的处理，建议传入 2sps 处理后降采样的序列或者在内部处理)
%                    *注意*: 为了与 FFE/VNLE 接口保持一致且简化逻辑，本函数假设 xRx 已经是经过同步且截断好的序列。
%                    如果是 2sps，本函数内部会进行滑动窗口采样。如果是 T-spaced FNN，请确保输入长度匹配。
%                    **这里采用 T-spaced 结构 (即输入窗对应 1 sps 的符号)**，如果 xRx 是 2sps，请在外部降采样或此处处理。
%                    **修正**: 参考 FFE_2pscenter，输入 xRx 是 2sps 的。FNN 通常做 T-spaced 均衡比较简单。
%                    为了获得最佳性能，我们构建一个 Fractionally Spaced FNN (输入是 2sps 的窗)。
%
%   xTx:             发送参考信号 (1 sps)
%   NumPreamble_TDE: 训练所用的符号数
%   InputLength:     输入层窗口长度 (同 FFE 的 Tap 数，例如 51, 101)
%   HiddenSize:      隐藏层神经元数量 (例如 [64, 32] 或单个 40)
%   LearningRate:    学习率 (例如 0.001)
%   MaxEpochs:       最大迭代次数
%
% 输出:
%   ye:              均衡后的输出信号 (1 sps)
%   net:             训练好的网络对象

    %% 1. 参数默认值与校验
    if nargin < 4, InputLength = 101; end 
    if nargin < 5, HiddenSize = 64; end   
    if nargin < 6, LearningRate = 0.0005; end
    if nargin < 7, MaxEpochs = 50; end
    if nargin < 8 || isempty(DelayCandidates), DelayCandidates = -30:30; end
    if nargin < 9 || isempty(OffsetCandidates), OffsetCandidates = [1 2]; end
    if nargin < 10 || isempty(ScanEpochs), ScanEpochs = min(5, MaxEpochs); end
    if nargin < 11 || isempty(ScanTrainSamples), ScanTrainSamples = min(5000, NumPreamble_TDE); end
    if nargin < 12 || isempty(ScanValSamples), ScanValSamples = min(2000, max(0, NumPreamble_TDE - ScanTrainSamples)); end

    % 检查是否可以使用 GPU
    useGPU = canUseGPU() && (exist('gpuArray', 'file') == 2);
    if useGPU
        disp('    [FNN] GPU Acceleration Detected: Enabled (CUDA).');
        execEnv = 'gpu';
    else
        disp('    [FNN] GPU not detected or toolbox missing. Using CPU.');
        execEnv = 'cpu';
    end

    %% 2. 数据构造 (Sliding Window Generation)
    
    Rx_Data = xRx(:);
    Tx_Data = xTx(:);
    
    % 记录 Target 的统计特性以便后续还原   
    y_mean = mean(Tx_Data);
    y_std  = std(Tx_Data);
    
    % 归一化 (Input 必须归一化，Output 归一化有助于训练收敛)
    Rx_Data = (Rx_Data - mean(Rx_Data)) / std(Rx_Data);
    Tx_Data = (Tx_Data - y_mean) / y_std;
    disp('    [FNN] Constructing Data Matrices...');

    % Delay/offset scan (Fast Linear Probe)
    best_mse = inf;
    best_delay = DelayCandidates(1);
    best_offset = OffsetCandidates(1);

    for oi = 1:length(OffsetCandidates)
        for di = 1:length(DelayCandidates)
            offset = OffsetCandidates(oi);
            delay = DelayCandidates(di);

            [X_scan, Y_scan] = build_dataset(Rx_Data, Tx_Data, InputLength, offset, delay, ScanTrainSamples + ScanValSamples);
            num_scan = size(X_scan, 2);
            if num_scan < (ScanTrainSamples + max(1, ScanValSamples))
                continue;
            end

            % Transpose to [Samples x Features]
            X_scan_t = X_scan.';
            Y_scan_t = Y_scan.';

            X_train = X_scan_t(1:ScanTrainSamples, :);
            Y_train = Y_scan_t(1:ScanTrainSamples, :);
            
            % --- Fast Linear Probe ---
            % Solve w = X \ y to instantly check correlation
            if size(X_train, 1) > size(X_train, 2)
                w_lin = X_train \ Y_train;
                
                % Check MSE
                if ScanValSamples > 0
                    X_val = X_scan_t(ScanTrainSamples + 1:ScanTrainSamples + ScanValSamples, :);
                    Y_val = Y_scan_t(ScanTrainSamples + 1:ScanTrainSamples + ScanValSamples, :);
                    Y_pred = X_val * w_lin;
                    mse_val = mean((Y_pred - Y_val).^2);
                else
                    Y_pred = X_train * w_lin;
                    mse_val = mean((Y_pred - Y_train).^2);
                end
            else
                mse_val = inf; 
            end

            if mse_val < best_mse
                best_mse = mse_val;
                best_delay = delay;
                best_offset = offset;
            end
        end
    end

    disp(['    [FNN] Selected Offset = ', num2str(best_offset), ', Delay = ', num2str(best_delay), ', Scan MSE = ', num2str(best_mse, '%.4g')]);

    [X, Y] = build_dataset(Rx_Data, Tx_Data, InputLength, best_offset, best_delay, []);
    num_valid = size(X, 2);
    
    %% 3. 数据集切分 (Train / Test)
    num_train = min(NumPreamble_TDE, num_valid);
    
    % 转置为 trainNetwork 所需格式 [N x Features]
    X_Transposed = X.'; 
    Y_Transposed = Y.'; 
    
    X_Train = X_Transposed(1:num_train, :);
    Y_Train = Y_Transposed(1:num_train, :);
    
    X_Test  = X_Transposed; 

    %% 4. 网络构建 (Network Architecture)
    
    disp('    [FNN] Configuring Network Layers (2-Layer MLP)...');
    
    layers = [
        featureInputLayer(InputLength, 'Normalization', 'none', 'Name', 'input')
        
        fullyConnectedLayer(HiddenSize, 'Name', 'fc1')
        tanhLayer('Name', 'tanh1') 
        
        fullyConnectedLayer(32, 'Name', 'fc2') % 新增第二隐层
        tanhLayer('Name', 'tanh2')
        
        fullyConnectedLayer(1, 'Name', 'output')
        regressionLayer('Name', 'loss')
    ];

    %% 5. 训练配置
    options = trainingOptions('adam', ...
        'MaxEpochs', MaxEpochs, ...
        'MiniBatchSize', 2048, ...
        'InitialLearnRate', LearningRate, ...
        'Shuffle', 'every-epoch', ...
        'Plots', 'none', ...
        'Verbose', 0, ...
        'ExecutionEnvironment', execEnv);
        
    %% 6. 开始训练
    disp(['    [FNN] Training Start (Samples: ', num2str(num_train), ', Epochs: ', num2str(MaxEpochs), ')...']);
    tic;
    net = trainNetwork(X_Train, Y_Train, layers, options);
    train_time = toc;
    disp(['    [FNN] Training Finished in ', num2str(train_time, '%.2f'), 's.']);
    
    %% 7. 推理与反归一化
    disp('    [FNN] Running Inference on Full Sequence...');
    
    % 再次获取全量数据集的索引，确保对齐
    [X_Full, ~, valid_tx_indices] = build_dataset(Rx_Data, Tx_Data, InputLength, best_offset, best_delay, []);
    X_Test = X_Full.';
    
    ye_predicted = predict(net, X_Test, 'MiniBatchSize', 8192, 'ExecutionEnvironment', execEnv);
    ye_predicted = double(ye_predicted'); % [1 x N_valid]
    
    % 反归一化
    ye_val = ye_predicted * y_std + y_mean;
    
    % 对齐输出: 创建与 Tx_Data 等长的数组，将预测值填入对应的位置
    ye = zeros(length(Tx_Data), 1);
    ye(valid_tx_indices) = ye_val;
    
    % 注意: 由于 ye 是列向量，如果原程序期望行向量，可能需要转置，
    % 但根据 FFE 代码习惯，通常列向量通用。这里保持列向量。
    
end

function [X, Y, tx_idx_out] = build_dataset(Rx_Data, Tx_Data, InputLength, offset, delay, max_samples)
    max_sym_rx = floor((length(Rx_Data) - InputLength - (offset - 1)) / 2) + 1;
    if max_sym_rx < 1
        X = zeros(InputLength, 0, 'single');
        Y = zeros(1, 0, 'single');
        tx_idx_out = [];
        return;
    end

    sym_idx = (1:max_sym_rx)';
    tx_idx = sym_idx + delay;
    valid_mask = tx_idx >= 1 & tx_idx <= length(Tx_Data);
    sym_idx = sym_idx(valid_mask);
    tx_idx = tx_idx(valid_mask);

    if ~isempty(max_samples)
        keep = min(max_samples, length(sym_idx));
        sym_idx = sym_idx(1:keep);
        tx_idx = tx_idx(1:keep);
    end
    
    tx_idx_out = tx_idx; % 输出有效的 Tx 索引

    start_indices = (sym_idx - 1) * 2 + offset;

    num_valid = length(start_indices);
    X = zeros(InputLength, num_valid, 'single');
    for i = 1:InputLength
        X(i, :) = Rx_Data(start_indices + (i-1));
    end
    Y = single(Tx_Data(tx_idx)).';
end

function flag = canUseGPU()
    try
        d = gpuDevice;
        flag = d.Index > 0;
    catch
        flag = false;
    end
end