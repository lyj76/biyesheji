function [ye, net, valid_tx_indices, best_delay, best_offset] = RNN_Implementation( ...
    xRx, xTx, NumPreamble_TDE, InputLength, HiddenSize, LearningRate, MaxEpochs, k, ...
    DelayCandidates, OffsetCandidates, ScanTrainSamples, ScanValSamples, ForceDelay, ForceOffset)
% RNN_Implementation (AR-RNN style MLP with hard-decision feedback)
%
% 输入：xRx (2sps), xTx (1sps), NumPreamble_TDE 训练符号数
% 输出：ye (与 xTx 等长；无效处为0), net, valid_tx_indices
%
% 关键：
% - 对齐：scan offset/delay (linear probe)
% - 训练：teacher forcing (feedback 用真实 Tx_n)
% - 推理：free-running + 手写前向传播 + 硬判决反馈
% - 硬判决电平：从训练标签 Yall 用 kmeans 自动估计，避免尺度错导致 BER=0.5

    %% ===== defaults =====
    if nargin < 4 || isempty(InputLength), InputLength = 101; end
    if nargin < 5 || isempty(HiddenSize), HiddenSize = 64; end
    if nargin < 6 || isempty(LearningRate), LearningRate = 1e-3; end
    if nargin < 7 || isempty(MaxEpochs), MaxEpochs = 50; end
    if nargin < 8 || isempty(k), k = 2; end
    if nargin < 9 || isempty(DelayCandidates), DelayCandidates = -30:30; end
    if nargin < 10 || isempty(OffsetCandidates), OffsetCandidates = [1 2]; end
    if nargin < 11 || isempty(ScanTrainSamples), ScanTrainSamples = min(5000, NumPreamble_TDE); end
    if nargin < 12 || isempty(ScanValSamples), ScanValSamples = min(2000, max(0, NumPreamble_TDE - ScanTrainSamples)); end
    if nargin < 13, ForceDelay = []; end
    if nargin < 14, ForceOffset = []; end

    if mod(InputLength,2)==0
        warning('[AR-RNN] InputLength even; recommend odd for centered window.');
    end

    %% ===== GPU detect (for training only) =====
    execEnv = 'cpu';
    try
        d = gpuDevice;
        if d.DeviceAvailable
            execEnv = 'gpu';
            % disp('    [AR-RNN] GPU Acceleration: Enabled (CUDA).');
        end
    catch
        % disp('    [AR-RNN] GPU not available. Using CPU.');
    end

    %% ===== preprocess & normalize =====
    Rx = xRx(:);
    Tx = xTx(:);

    y_mean = mean(Tx);
    y_std  = std(Tx);

    Rx = (Rx - mean(Rx)) / std(Rx);
    Tx_n = (Tx - y_mean) / y_std;

    %% ===== 1) scan offset/delay (linear probe) =====
    % disp('    [AR-RNN] Scanning best offset/delay using Linear Probe...');

    best_ser = inf;
    best_mse = inf;
    best_delay = DelayCandidates(1);
    best_offset = OffsetCandidates(1);

    % --- fixed PAM4 levels from Tx_n (robust for scan) ---
    Yref = double(Tx_n(1:min(NumPreamble_TDE, length(Tx_n))));
    [~, Cref] = kmeans(Yref(:), 4, 'Replicates', 3);
    levels_ref = sort(Cref(:)).';
    thr_ref = (levels_ref(1:3) + levels_ref(2:4))/2;

    for oi = 1:numel(OffsetCandidates)
        offset = OffsetCandidates(oi);
        for di = 1:numel(DelayCandidates)
            delay = DelayCandidates(di);

            [Xscan, Yscan] = build_center_window_dataset(Rx, Tx_n, InputLength, offset, delay, ScanTrainSamples + ScanValSamples);
            if size(Xscan,2) < (ScanTrainSamples + max(1,ScanValSamples))
                continue;
            end

            Xs = Xscan.';  % [N x F]
            Ys = Yscan.';  % [N x 1]

            Xtr = Xs(1:ScanTrainSamples,:);
            Ytr = Ys(1:ScanTrainSamples,:);

            if size(Xtr,1) <= size(Xtr,2)
                continue;
            end

            w = Xtr \ Ytr;

            if ScanValSamples > 0
                Xva = Xs(ScanTrainSamples+1:ScanTrainSamples+ScanValSamples,:);
                Yva = Ys(ScanTrainSamples+1:ScanTrainSamples+ScanValSamples,:);
                Yp  = Xva * w;
            else
                Xva = Xtr;
                Yva = Ytr;
                Yp  = Xtr * w;
            end

            mse = mean((Yp - Yva).^2);

            % --- decision-domain metric: SER using fixed PAM4 thresholds ---
            Yp_q = hard_slice_pam4(Yp, levels_ref, thr_ref);
            Yv_q = hard_slice_pam4(Yva, levels_ref, thr_ref);
            ser = mean(Yp_q ~= Yv_q);

            if ser < best_ser
                best_ser = ser;
                best_mse = mse;
                best_delay = delay;
                best_offset = offset;
            end
        end
    end

    % disp(['    [AR-RNN] Selected Offset=',num2str(best_offset), ...
    %       ', Delay=',num2str(best_delay),', SER=',num2str(best_ser,'%.4g'), ...
    %       ', Linear MSE=',num2str(best_mse,'%.4g')]);

    if ~isempty(ForceDelay)
        % disp(['    [AR-RNN] ForceDelay override: ', num2str(best_delay), ' -> ', num2str(ForceDelay)]);
        best_delay = ForceDelay;
    end
    if ~isempty(ForceOffset)
        % disp(['    [AR-RNN] ForceOffset override: ', num2str(best_offset), ' -> ', num2str(ForceOffset)]);
        best_offset = ForceOffset;
    end

    %% ===== 2) build full aligned dataset =====
    [Xall, Yall, valid_tx_indices] = build_center_window_dataset(Rx, Tx_n, InputLength, best_offset, best_delay, []);
    Nvalid = size(Xall,2);
    if Nvalid < (k+10)
        error('[AR-RNN] Not enough valid samples. Check InputLength/offset/delay.');
    end

    Ntrain = min(NumPreamble_TDE, Nvalid);

    %% ===== 3) training set (teacher forcing) =====
    start_n = k + 1;
    end_n   = Ntrain;

    Xar = zeros(InputLength + k, end_n - start_n + 1, 'single');
    Yar = zeros(1, end_n - start_n + 1, 'single');

    idx = 1;
    for n = start_n:end_n
        fb = Yall(1, n-1 : -1 : n-k);       % [1 x k] from true label
        Xar(:,idx) = [Xall(:,n); fb.'];     % (InputLength+k) x 1
        Yar(1,idx) = Yall(1,n);
        idx = idx + 1;
    end

    X_Train = Xar.';   % [Ns x F]
    Y_Train = Yar.';   % [Ns x 1]

    %% ===== 4) network (MLP) =====
    % disp('    [AR-RNN] Configuring AR-MLP Network (Eq.(2) style)...');

    layers = [
        featureInputLayer(InputLength + k, 'Normalization','none', 'Name','input')

        fullyConnectedLayer(HiddenSize, 'Name','fc1')
        tanhLayer('Name','tanh1')

        fullyConnectedLayer(32, 'Name','fc2')
        tanhLayer('Name','tanh2')

        fullyConnectedLayer(1, 'Name','out')
        regressionLayer('Name','loss')
    ];

    options = trainingOptions('adam', ...
         'L2Regularization', 1e-3,...
        'MaxEpochs', 50, ...
        'MiniBatchSize', 512, ...
        'InitialLearnRate', LearningRate, ...
        'Shuffle','every-epoch', ...
        'Verbose', 0, ...
        'Plots','none', ...
        'ExecutionEnvironment', execEnv);

    %% ===== 5) train =====
    % disp(['    [AR-RNN] Training Start (Samples=',num2str(size(X_Train,1)), ...
    %       ', Epochs=',num2str(MaxEpochs), ')...']);
    tic;
    net = trainNetwork(X_Train, Y_Train, layers, options);
    % disp(['    [AR-RNN] Training Finished in ', num2str(toc,'%.3f'), ' s']);

    %% ===== 6) inference: free-running, manual forward + HARD decision feedback =====
    % disp('    [AR-RNN] Inference on Full Sequence (free-running, manual forward + hard feedback)...');

    % ---- extract weights by name (robust) ----
    L = net.Layers;
    l_fc1 = L(strcmp({L.Name}, 'fc1'));
    l_fc2 = L(strcmp({L.Name}, 'fc2'));
    l_out = L(strcmp({L.Name}, 'out'));

    W1 = gather(l_fc1.Weights); b1 = gather(l_fc1.Bias);
    W2 = gather(l_fc2.Weights); b2 = gather(l_fc2.Bias);
    W3 = gather(l_out.Weights); b3 = gather(l_out.Bias);

    % ---- estimate 4 PAM levels from training labels (normalized domain) ----
    Ytr = double(Yall(1, 1:Ntrain)).';
    Ytr = Ytr(isfinite(Ytr));
    [~, C] = kmeans(Ytr, 4, 'Replicates', 5);
    pam4_levels = sort(C(:)).';                 % 1x4
    thr = (pam4_levels(1:3) + pam4_levels(2:4))/2; % 1x3
    % disp(['    [AR-RNN] learned PAM4 levels (norm) = ', mat2str(pam4_levels,4)]);

    tanhf = @(z) tanh(z);

    ye_n  = zeros(Nvalid,1);   % continuous output (norm)
    ye_fb = zeros(Nvalid,1);   % hard feedback levels (norm)

    ye_n(1:k)  = Yall(1,1:k).';
    ye_fb(1:k) = hard_slice_pam4(ye_n(1:k), pam4_levels, thr);

    for n = (k+1):Nvalid
        fb = ye_fb(n-1:-1:n-k);               % use hard feedback
        u  = [Xall(:,n); single(fb)];
        u  = double(u);

        h1 = tanhf(W1*u + b1);
        h2 = tanhf(W2*h1 + b2);
        yhat = W3*h2 + b3;

        ye_n(n)  = yhat;
        ye_fb(n) = hard_slice_pam4(yhat, pam4_levels, thr);
    end

    % ---- de-normalize back to original Tx amplitude ----
    ye_val = ye_n * y_std + y_mean;

    ye = zeros(length(Tx),1);
    ye(valid_tx_indices) = ye_val;

end

%% ================= helper: build centered window dataset (2sps) =================
function [X, Y, tx_idx_out] = build_center_window_dataset(Rx_Data, Tx_Data, InputLength, offset, delay, max_samples)
    HalfLen = floor(InputLength/2);

    max_sym_rx = floor((length(Rx_Data) - 1 - offset)/2) + 1;
    if max_sym_rx < 1
        X = zeros(InputLength,0,'single');
        Y = zeros(1,0,'single');
        tx_idx_out = [];
        return;
    end

    sym_idx = (1:max_sym_rx).';
    tx_idx  = sym_idx + delay;

    valid_mask = (tx_idx >= 1) & (tx_idx <= length(Tx_Data));
    sym_idx = sym_idx(valid_mask);
    tx_idx  = tx_idx(valid_mask);

    if isempty(sym_idx)
        X = zeros(InputLength,0,'single');
        Y = zeros(1,0,'single');
        tx_idx_out = [];
        return;
    end

    center = (sym_idx - 1) * 2 + offset;
    start_i = center - HalfLen;
    end_i   = center + HalfLen;

    valid2 = (start_i >= 1) & (end_i <= length(Rx_Data));
    sym_idx = sym_idx(valid2);
    tx_idx  = tx_idx(valid2);
    center  = center(valid2);

    if isempty(sym_idx)
        X = zeros(InputLength,0,'single');
        Y = zeros(1,0,'single');
        tx_idx_out = [];
        return;
    end

    if ~isempty(max_samples)
        keep = min(max_samples, length(sym_idx));
        sym_idx = sym_idx(1:keep);
        tx_idx  = tx_idx(1:keep);
        center  = center(1:keep);
    end

    N = length(sym_idx);
    X = zeros(InputLength, N, 'single');
    for n = 1:N
        idx = (center(n)-HalfLen) : (center(n)+HalfLen);
        X(:,n) = single(Rx_Data(idx));
    end

    Y = single(Tx_Data(tx_idx)).';
    tx_idx_out = tx_idx;
end

%% ================= helper: hard slice to 4 levels =================
function yq = hard_slice_pam4(y, levels, thr)
    y = double(y);
    yq = zeros(size(y));

    yq(y < thr(1)) = levels(1);
    yq(y >= thr(1) & y < thr(2)) = levels(2);
    yq(y >= thr(2) & y < thr(3)) = levels(3);
    yq(y >= thr(3)) = levels(4);
end
