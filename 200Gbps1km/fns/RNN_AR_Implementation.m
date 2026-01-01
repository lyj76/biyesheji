function [ye, net, valid_tx_indices, best_delay, best_offset] = RNN_AR_Implementation( ...
    xRx, xTx, NumPreamble_TDE, InputLength, HiddenSize, LearningRate, MaxEpochs, k, ...
    DelayCandidates, OffsetCandidates)
% RNN_AR_Implementation (Auto-Regressive MLP with Soft Feedback)
%   Structure: Residual Network with SOFT Output Feedback
%   Contrast: WD-RNN uses HARD Decision Feedback.
%
%   Input: [Rx_Window; Soft_Output(n-1...n-k)]
%   Output: Residual correction to Linear FFE

    %% 0) Defaults
    if nargin < 4 || isempty(InputLength), InputLength = 101; end
    if nargin < 5 || isempty(HiddenSize), HiddenSize = 32; end
    if nargin < 6 || isempty(LearningRate), LearningRate = 1e-3; end
    if nargin < 7 || isempty(MaxEpochs), MaxEpochs = 30; end
    if nargin < 8 || isempty(k), k = 5; end % AR typically uses shorter history than WD
    if nargin < 9 || isempty(DelayCandidates), DelayCandidates = -20:20; end
    if nargin < 10 || isempty(OffsetCandidates), OffsetCandidates = [1 2]; end

    execEnv = 'auto';
    try, if canUseGPU_local() && exist('gpuArray','file')==2, execEnv = 'gpu'; end; catch, execEnv='cpu'; end

    %% 1) Preprocess
    Rx = xRx(:); Tx = xTx(:);
    y_mean = mean(Tx); y_std  = std(Tx);
    Rx = (Rx - mean(Rx)) / std(Rx);
    Tx_n = (Tx - y_mean) / y_std;

    %% 2) Linear Probe (FFE Baseline)
    % Use short feedback history for linear probe to avoid instability
    ScanSamples = min(4000, NumPreamble_TDE);
    best_mse = inf; best_delay = 0; best_offset = OffsetCandidates(1); 
    w_lin = zeros(InputLength + k, 1); % Init with zeros to be safe
    
    for off = OffsetCandidates
        for d = DelayCandidates
            % Use Teacher Forcing for probe
            [X, Y] = build_data(Rx, Tx_n, InputLength, k, off, d, ScanSamples);
            if size(X,2)<500, continue; end
            X=double(X.'); Y=double(Y.');
            R = X'*X; w = (R + 1e-3*eye(size(R))) \ (X'*Y);
            mse = mean((X*w - Y).^2);
            if mse < best_mse, best_mse=mse; best_delay=d; best_offset=off; w_lin=w; end
        end
    end

    %% 3) Train NN (Residuals using Teacher Forcing)
    [X_all, Y_all, valid_tx_indices] = build_data(Rx, Tx_n, InputLength, k, best_offset, best_delay, NumPreamble_TDE);
    X_tr = X_all.'; Y_tr = Y_all.';
    
    % Linear Pred & Residual
    Y_lin = X_tr * w_lin;
    Y_res = single(Y_tr - Y_lin);
    
    layers = [
        featureInputLayer(InputLength+k, 'Normalization','none')
        fullyConnectedLayer(HiddenSize, 'WeightsInitializer','he')
        tanhLayer
        fullyConnectedLayer(ceil(HiddenSize/2), 'WeightsInitializer','he')
        tanhLayer
        fullyConnectedLayer(1)
        regressionLayer
    ];
    
    opts = trainingOptions('adam', 'MaxEpochs', MaxEpochs, 'MiniBatchSize', 512, ...
        'InitialLearnRate', LearningRate, 'L2Regularization', 1e-4, ...
        'Shuffle','every-epoch', 'Verbose',0, 'Plots','none', 'ExecutionEnvironment', execEnv);
    
    net = trainNetwork(X_tr, Y_res, layers, opts);

    %% 4) Inference (Soft Feedback Loop)
    [Rx_Mat, ~, all_idx] = build_rx_only(Rx, Tx_n, InputLength, best_offset, best_delay);
    
    % DEBUG PRINT
    % disp(['RNN_AR Debug: Delay=', num2str(best_delay), ' Offset=', num2str(best_offset), ' Valid=', num2str(length(all_idx))]);
    
    N_tot = size(Rx_Mat, 2);
    ye_tot = zeros(N_tot, 1);
    
    % Extract Weights
    W1=gather(net.Layers(2).Weights); b1=gather(net.Layers(2).Bias);
    W2=gather(net.Layers(4).Weights); b2=gather(net.Layers(4).Bias);
    W3=gather(net.Layers(6).Weights); b3=gather(net.Layers(6).Bias);
    
    % Soft Feedback Buffer
    % Init with mean value (0) or known preamble
    fb = zeros(k,1); 
    if k>0 && ~isempty(Y_all), fb = double(Y_all(end-k+1:end)).'; end
    
    for n = 1:N_tot
        in_vec = [Rx_Mat(:,n); fb];
        
        % Linear + Residual
        try
            y_lin = w_lin.' * double(in_vec);
        catch ME
            disp(['Error in Matrix Mul: w_lin size = ', mat2str(size(w_lin)), ...
                  ', in_vec size = ', mat2str(size(in_vec))]);
            rethrow(ME);
        end
        h1 = tanh(W1*in_vec + b1);
        h2 = tanh(W2*h1 + b2);
        y_res = W3*h2 + b3;
        
        y_out = y_lin + y_res;
        ye_tot(n) = y_out;
        
        % SOFT Feedback: Pass y_out directly back
        % (Optionally saturate to avoid explosion)
        if k > 0
            y_soft = max(-3, min(3, y_out)); % Clip slightly to typical PAM4 range
            fb = [y_soft; fb(1:end-1)];
        end
    end

    ye = zeros(length(Tx),1); ye(all_idx) = ye_tot * y_std + y_mean;
    
    % --- SELF CHECK DEBUG ---
    % Calculate BER internally to verify "0 error" claim
    test_mask = all_idx > NumPreamble_TDE;
    if any(test_mask)
        y_test_raw = ye(all_idx(test_mask));
        tx_test_raw = Tx(all_idx(test_mask));
        
        % Simple PAM4 Demod
        [~, centers] = kmeans(double(y_test_raw(1:min(5000,end))), 4, 'Replicates',3);
        lvls = sort(centers);
        thr = (lvls(1:3)+lvls(2:4))/2;
        y_hard = zeros(size(y_test_raw));
        y_hard(y_test_raw < thr(1)) = 0;
        y_hard(y_test_raw >= thr(1) & y_test_raw < thr(2)) = 1;
        y_hard(y_test_raw >= thr(2) & y_test_raw < thr(3)) = 2;
        y_hard(y_test_raw >= thr(3)) = 3;
        
        % Map Tx (assuming Tx is already 0,1,2,3 or needs mapping)
        % Our Tx is amplitude (-3,-1,1,3). Need to map to 0,1,2,3 for biterr
        % Simple way: sort unique values
        uTx = unique(tx_test_raw);
        if length(uTx) == 4
             map_tx = zeros(size(tx_test_raw));
             for i=1:4, map_tx(tx_test_raw == uTx(i)) = i-1; end
             [~, ber_check] = biterr(y_hard, map_tx, 2);
             % disp(['    [RNN_AR Internal Check] Test BER = ', num2str(ber_check)]);
        end
    end
end

function [X, Y, idx] = build_data(Rx, Tx, L, k, off, d, max_n)
    st = k+1; ed = length(Tx);
    if ~isempty(max_n), ed = min(ed, st+max_n-1); end
    n = (st:ed).';
    
    center = 2*(n+d)+off;
    st_i = center - floor(L/2);
    valid = (st_i >= 1) & (center+floor(L/2) <= length(Rx));
    n = n(valid); st_i = st_i(valid);
    
    N = length(n);
    X = zeros(L+k, N, 'single');
    Y = single(Tx(n));
    
    for i=1:N
        rx_p = Rx(st_i(i) : st_i(i)+L-1);
        fb_p = Tx(n(i)-1 : -1 : n(i)-k); % True Tx (Teacher Forcing)
        X(:,i) = [rx_p; fb_p];
    end
    idx = n;
    Y = Y.'; % [1 x N]
end

function [RxMat, Tx, idx] = build_rx_only(Rx, Tx, L, off, d)
    n = (1:length(Tx)).';
    center = 2*(n+d)+off;
    st_i = center - floor(L/2);
    valid = (st_i>=1) & (center+floor(L/2)<=length(Rx));
    n = n(valid); st_i = st_i(valid);
    
    N = length(n);
    RxMat = zeros(L, N, 'single');
    for i=1:N, RxMat(:,i) = Rx(st_i(i):st_i(i)+L-1); end
    idx = n; Tx = [];
end

function f = canUseGPU_local(), try, d=gpuDevice; f=d.Index>0; catch, f=false; end, end