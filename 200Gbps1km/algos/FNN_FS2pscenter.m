function [ye, net, valid_tx_indices, best_delay, best_phase] = FNN_FS2pscenter( ...
    xRx_2sps, xTx_1sps, NumPreamble_TDE, TapLen, HiddenSize, LearningRate, MaxEpochs, DelayCandidates, PhaseCandidates, init_net)

% FNN_FS2pscenter (Residual Version):
%   Ensures performance >= Linear FFE by explicitly solving the linear part first.
%   Structure: Output = Linear_Filter(Input) + MLP(Input)

    %% 0) Defaults
    if nargin < 4 || isempty(TapLen), TapLen = 111; end
    if nargin < 5 || isempty(HiddenSize), HiddenSize = 48; end % Smaller hidden size for residual
    if nargin < 6 || isempty(LearningRate), LearningRate = 1e-3; end
    if nargin < 7 || isempty(MaxEpochs), MaxEpochs = 40; end
    if nargin < 8 || isempty(DelayCandidates), DelayCandidates = -60:60; end
    if nargin < 9 || isempty(PhaseCandidates), PhaseCandidates = [0 1]; end
    if nargin < 10, init_net = []; end

    if mod(TapLen,2) == 0
        error('TapLen must be odd (e.g., 111).');
    end

    execEnv = 'auto';
    try
        if canUseGPU_local() && exist('gpuArray','file')==2
            execEnv = 'gpu';
        end
    catch
        execEnv = 'cpu';
    end

    %% 1) Preprocess
    rx = xRx_2sps(:);
    tx = xTx_1sps(:);
    rx_scale = mean(abs(rx)) + eps;
    tx_scale = mean(abs(tx)) + eps;
    rx_n = rx / rx_scale;
    tx_n = tx / tx_scale;
    Half = (TapLen - 1)/2;
    Padding = Half;
    rx_pad = [zeros(Padding,1); rx_n; zeros(Padding,1)];

    %% 2) Linear Probe (Find best delay & Linear Weights)
    ProbeLen = min(NumPreamble_TDE, length(tx_n));
    best_mse = inf;
    best_delay = 0;
    best_phase = PhaseCandidates(1);
    w_lin = zeros(TapLen, 1);

    for ph = PhaseCandidates
        for d = DelayCandidates
            [X_probe, Y_probe, ~] = build_fs_dataset(rx_pad, tx_n, TapLen, d, ph, ProbeLen, Padding);
            if isempty(Y_probe), continue; end

            Xp = double(X_probe.');   % [N x TapLen]
            Yp = double(Y_probe(:));  % [N x 1]

            % Ridge Regression (Linear FFE Solution)
            R = (Xp.'*Xp + 1e-5*eye(size(Xp,2)));
            w = R \ (Xp.'*Yp);
            mse = mean((Xp*w - Yp).^2);

            if mse < best_mse
                best_mse = mse;
                best_delay = d;
                best_phase = ph;
                w_lin = w;
            end
        end
    end

    %% 3) Build Train/Val set (Residuals)
    [X_all, Y_all, ~] = build_fs_dataset(rx_pad, tx_n, TapLen, best_delay, best_phase, NumPreamble_TDE, Padding);

    X = X_all.';            % [N x TapLen]
    Y_true = Y_all(:);      % [N x 1]
    
    % Calculate Linear Prediction
    Y_lin = double(X) * w_lin;
    
    % *** RESIDUAL TARGET ***
    Y_res = single(Y_true - Y_lin); 
    
    % Shuffle + split
    Ntot = size(X,1);
    perm = randperm(Ntot);
    X = X(perm,:);
    Y_res = Y_res(perm,:);

    Ntr = floor(0.9*Ntot);
    Xtr = single(X(1:Ntr,:));
    Ytr = Y_res(1:Ntr,:);
    Xva = single(X(Ntr+1:end,:));
    Yva = Y_res(Ntr+1:end,:);

    %% 4) Train NN on Residuals
    if isempty(init_net)
        layers = [
            featureInputLayer(TapLen, 'Normalization','none', 'Name','in')
            fullyConnectedLayer(HiddenSize, 'Name','fc1', 'WeightsInitializer','he')
            tanhLayer('Name','act1') % Tanh is better for residuals centered at 0
            fullyConnectedLayer(ceil(HiddenSize/2), 'Name','fc2', 'WeightsInitializer','he')
            tanhLayer('Name','act2')
            fullyConnectedLayer(1, 'Name','out')
            regressionLayer('Name','loss')
        ];
    else
        layers = init_net.Layers;
    end

    opts = trainingOptions('adam', ...
        'MaxEpochs', MaxEpochs, ...
        'MiniBatchSize', 256, ...
        'InitialLearnRate', LearningRate, ...
        'L2Regularization', 1e-3, ...
        'ValidationData', {Xva, Yva}, ...
        'ValidationPatience', 6, ...
        'Shuffle','every-epoch', ...
        'Verbose', 0, ...
        'Plots','none', ...
        'ExecutionEnvironment', execEnv);

    net = trainNetwork(Xtr, Ytr, layers, opts);

    %% 5) Inference (Linear + Residual)
    [X_full, ~, valid_tx_indices] = build_fs_dataset(rx_pad, tx_n, TapLen, best_delay, best_phase, [], Padding);

    Xte = X_full.'; 
    
    % 1. Linear Part
    y_lin_full = double(Xte) * w_lin;
    
    % 2. Residual Part
    y_res_full = predict(net, Xte, 'MiniBatchSize', 8192, 'ExecutionEnvironment', execEnv);
    y_res_full = double(y_res_full(:));

    % 3. Combine
    yhat_n = y_lin_full + y_res_full;

    % Rescale
    ye = yhat_n * tx_scale;
    valid_tx_indices = valid_tx_indices(:);
end

%% ===== helper =====
function [X, Y, tx_idx] = build_fs_dataset(rx_pad, tx_1sps, TapLen, delay_samp, phase, max_syms, Padding)
    Nsym = length(tx_1sps);
    n = (1:Nsym).';
    center = (2*n) + delay_samp + phase + Padding;
    Half = (TapLen-1)/2;
    st = center - Half;
    ed = center + Half;
    valid = (st >= 1) & (ed <= length(rx_pad));
    n_valid = n(valid);
    st_valid = st(valid);

    if isempty(n_valid), X=[]; Y=[]; tx_idx=[]; return; end
    if ~isempty(max_syms)
        K = min(length(n_valid), max_syms);
        n_valid = n_valid(1:K);
        st_valid = st_valid(1:K);
    end

    K = length(n_valid);
    X = zeros(TapLen, K, 'single');
    for i = 1:K
        s = st_valid(i);
        X(:,i) = single(rx_pad(s : s+TapLen-1));
    end
    Y = single(tx_1sps(n_valid));
    tx_idx = n_valid;
end

function flag = canUseGPU_local()
    try, d = gpuDevice; flag = d.Index > 0; catch, flag = false; end
end