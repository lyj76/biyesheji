function [ye, net, valid_tx_indices, best_delay, best_offset] = WDRNN_Implementation( ...
    xRx, xTx, NumPreamble_TDE, InputLength, HiddenSize, LearningRate, MaxEpochs, k, ...
    DelayCandidates, OffsetCandidates)
% WDRNN_Implementation - Physics-Informed Volterra WD-RNN
% 核心改动：
% 1. 回归单层结构 (Hidden=32)：保证 2.7万 数据量下的收敛性
% 2. 物理特征注入：输入不仅包含 x，还包含 x.^2 (模拟光强检测非线性)
% 3. WD 参数微调：回归论文推荐的 beta=0.14 (对 5.6dBm 更有效)

    %% ===== 0. 参数设置 =====
    if nargin < 4 || isempty(InputLength), InputLength = 61; end
    if nargin < 5 || isempty(HiddenSize), HiddenSize = 32; end % 单层给32个神经元够了
    if nargin < 6 || isempty(LearningRate), LearningRate = 1e-3; end
    if nargin < 7 || isempty(MaxEpochs), MaxEpochs = 60; end % 单层收敛快，60轮足够
    if nargin < 8 || isempty(k), k = 25; end
    if nargin < 9 || isempty(DelayCandidates), DelayCandidates = -20:20; end
    if nargin < 10 || isempty(OffsetCandidates), OffsetCandidates = [1 2]; end
    
    ScanTrainSamples = min(5000, NumPreamble_TDE);
    ScanValSamples = min(2000, max(0, NumPreamble_TDE - ScanTrainSamples));

    % WD 参数 (回归论文原值，高SNR下更准)
    alpha = 5;      
    beta = 0.06;    

    %% ===== 1. 预处理 =====
    Rx = xRx(:); Tx = xTx(:);
    y_mean = mean(Tx); y_std = std(Tx);
    
    % 归一化 Rx (Tx 保持原样或去均值，为了 K-means 准确)
    Rx = (Rx - mean(Rx)) / std(Rx);
    Tx_n = (Tx - y_mean) / y_std;

    %% ===== 2. 同步 (Linear Probe) =====
    best_ser = inf; best_delay = DelayCandidates(1); best_offset = OffsetCandidates(1);
    
    % 学习电平
    Yref = double(Tx_n(1:min(NumPreamble_TDE, length(Tx_n))));
    [~, Cref] = kmeans(Yref(:), 4, 'Replicates', 3);
    levels_ref = sort(Cref(:)).';
    thr_ref = (levels_ref(1:3) + levels_ref(2:4))/2;
    
    for oi = 1:numel(OffsetCandidates)
        offset = OffsetCandidates(oi);
        for di = 1:numel(DelayCandidates)
            delay = DelayCandidates(di);
            [Xscan, Yscan] = build_center_window_dataset(Rx, Tx_n, InputLength, offset, delay, ScanTrainSamples+ScanValSamples);
            if size(Xscan,2) < (ScanTrainSamples+10), continue; end
            Xs=Xscan.'; Ys=Yscan.';
            Xtr=Xs(1:ScanTrainSamples,:); Ytr=Ys(1:ScanTrainSamples,:);
            if size(Xtr,1) <= size(Xtr,2), continue; end
            w = Xtr \ Ytr; Yp = Xtr * w;
            Yp_q = hard_slice_pam4(Yp, levels_ref, thr_ref);
            Yv_q = hard_slice_pam4(Ytr, levels_ref, thr_ref);
            ser = mean(Yp_q ~= Yv_q);
            if ser < best_ser, best_ser=ser; best_delay=delay; best_offset=offset; end
        end
    end
    
    %% ===== 3. 构建数据集 =====
    [Xall, Yall, valid_tx_indices] = build_center_window_dataset(Rx, Tx_n, InputLength, best_offset, best_delay, []);
    Nvalid = size(Xall,2);
    Ntrain = min(NumPreamble_TDE, Nvalid);

    %% ===== 4. 特征工程 (关键改动：Physics Injection) =====
    % 构造增强特征：[原始输入, 原始输入的平方]
    % 这让单层网络瞬间拥有了拟合 Volterra 非线性的能力
    start_n = k + 1;
    end_n   = Ntrain;
    
    % 输入维度翻倍：(InputLength + k) * 2
    feat_dim = (InputLength + k) * 2;
    
    Xar = zeros(feat_dim, end_n - start_n + 1, 'single');
    Yar = zeros(1, end_n - start_n + 1, 'single');
    
    idx = 1;
    for n = start_n:end_n
        fb = Yall(1, n-1 : -1 : n-k);
        
        % 原始特征 vector
        raw_feat = [Xall(:,n); fb.'];
        
        % 物理特征 vector (平方项，模拟光电检测)
        % 注意：只对 Rx 部分做平方更有物理意义，但为了简单，全做也无妨
        poly_feat = raw_feat .^ 2; 
        
        % 拼接
        Xar(:,idx) = [raw_feat; poly_feat];
        Yar(1,idx) = Yall(1,n);
        idx = idx + 1;
    end
    X_Train = Xar.'; Y_Train = Yar.';

    %% ===== 5. 网络构建 (回归单层，但输入更强) =====
    layers = [
        featureInputLayer(feat_dim, 'Normalization','none', 'Name','input')
        
        % 单隐层 (宽一点，由20加到32)
        fullyConnectedLayer(HiddenSize, 'WeightsInitializer', 'he', 'Name', 'fc1')
        tanhLayer('Name', 'tanh1')
        
        % 输出
        fullyConnectedLayer(1, 'WeightsInitializer', 'he', 'Name', 'out')
        regressionLayer('Name','loss')
    ];
    
    % 学习率策略
    options = trainingOptions('adam', ...
        'MaxEpochs', MaxEpochs, ...
        'MiniBatchSize', 256, ...
        'InitialLearnRate', LearningRate, ...
        'L2Regularization', 1e-4, ...
        'Shuffle','every-epoch', ...
        'Verbose', 0, ...
        'ExecutionEnvironment', 'auto');

    %% ===== 6. 训练 =====
    net = trainNetwork(X_Train, Y_Train, layers, options);

    %% ===== 7. 推理 (适配物理特征) =====
    L = net.Layers;
    l_fc1 = L(strcmp({L.Name}, 'fc1'));
    if isempty(l_fc1), l_fc1 = L(2); else, l_fc1 = l_fc1(1); end
    l_out = L(strcmp({L.Name}, 'out'));
    if isempty(l_out), l_out = L(end-1); else, l_out = l_out(1); end

    W1 = gather(l_fc1.Weights); b1 = gather(l_fc1.Bias);
    W2 = gather(l_out.Weights); b2 = gather(l_out.Bias);

    % 学习电平
    Ytr_vals = double(Yall(1, 1:Ntrain)).';
    [~, C] = kmeans(Ytr_vals, 4, 'Replicates', 3);
    pam4_levels = sort(C(:)).'; 
    thr = (pam4_levels(1:3) + pam4_levels(2:4))/2;
    dist_levels = mean(diff(pam4_levels));
    
    ye_n = zeros(Nvalid, 1);
    ye_fb = zeros(Nvalid, 1);
    ye_n(1:k) = Yall(1,1:k).';
    ye_fb(1:k) = Yall(1,1:k).';
    
    tanhf = @(z) tanh(z);

    for n = (k+1):Nvalid
        % 1. 构造反馈
        fb = ye_fb(n-1:-1:n-k);
        
        % 2. 构造增强特征 (与训练时一致)
        raw_feat = [Xall(:,n); single(fb)];
        poly_feat = raw_feat .^ 2;
        u = [raw_feat; poly_feat]; % 维度翻倍
        
        u = double(u);
        
        % 3. 单层前向传播
        h1 = tanhf(W1*u + b1);
        y_soft = W2*h1 + b2;
        ye_n(n) = y_soft;
        
        % 4. WD 逻辑 (beta=0.14)
        y_hard = hard_slice_pam4(y_soft, pam4_levels, thr);
        abs_err = abs(y_soft - y_hard);
        
        gamma_n = 1 - abs_err / (dist_levels/2);
        if gamma_n < 0, gamma_n = 0; end
        
        term = -alpha * (gamma_n/beta - 1);
        S_val = 0.5 * ( (1 - exp(term)) / (1 + exp(term)) + 1 );
        
        ye_fb(n) = S_val * y_hard + (1 - S_val) * y_soft;
    end
    
    %% ===== 8. 输出 =====
    ye_val = ye_n * y_std + y_mean;
    ye = zeros(length(Tx), 1);
    ye(valid_tx_indices) = ye_val;
end

%% Helpers (保持不变)
function [X, Y, tx_idx_out] = build_center_window_dataset(Rx, Tx, InputLength, offset, delay, max_samples)
    HalfLen = floor(InputLength/2);
    max_sym = floor((length(Rx)-1-offset)/2)+1;
    if max_sym<1, X=[];Y=[];tx_idx_out=[]; return; end
    sym_idx=(1:max_sym).'; tx_idx=sym_idx+delay;
    mask=(tx_idx>=1)&(tx_idx<=length(Tx));
    sym_idx=sym_idx(mask); tx_idx=tx_idx(mask);
    center=(sym_idx-1)*2+offset;
    valid2=(center-HalfLen>=1)&(center+HalfLen<=length(Rx));
    sym_idx=sym_idx(valid2); tx_idx=tx_idx(valid2); center=center(valid2);
    if ~isempty(max_samples), k=min(max_samples,length(sym_idx)); sym_idx=sym_idx(1:k); tx_idx=tx_idx(1:k); center=center(1:k); end
    N=length(sym_idx); X=zeros(InputLength,N,'single');
    for n=1:N, X(:,n)=single(Rx(center(n)-HalfLen : center(n)+HalfLen)); end
    Y=single(Tx(tx_idx)).'; tx_idx_out=tx_idx;
end

function yq = hard_slice_pam4(y, levels, thr)
    y=double(y); yq=zeros(size(y));
    yq(y<thr(1))=levels(1);
    yq(y>=thr(1)&y<thr(2))=levels(2);
    yq(y>=thr(2)&y<thr(3))=levels(3);
    yq(y>=thr(3))=levels(4);
end