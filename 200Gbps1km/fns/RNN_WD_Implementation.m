function [ye, net, valid_tx_indices, best_delay, best_offset] = RNN_WD_Implementation( ...
    xRx, xTx, NumPreamble_TDE, InputLength, HiddenSize, LearningRate, MaxEpochs, k, ...
    DelayCandidates, OffsetCandidates)
% RNN_WD_Implementation (Windowed-Decision RNN)
%   Structure: Residual Network with HARD Decision Feedback
%   Input: [Rx_Window; Hard_Decisions(n-1...n-k)]
%   Output: Residual correction to Linear FFE

    %% 0) Defaults
    if nargin < 4 || isempty(InputLength), InputLength = 101; end
    if nargin < 5 || isempty(HiddenSize), HiddenSize = 32; end
    if nargin < 6 || isempty(LearningRate), LearningRate = 1e-3; end
    if nargin < 7 || isempty(MaxEpochs), MaxEpochs = 30; end
    if nargin < 8 || isempty(k), k = 25; end
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
    ScanSamples = min(4000, NumPreamble_TDE);
    best_mse = inf; best_delay = 0; best_offset = OffsetCandidates(1); w_lin = [];
    
    for off = OffsetCandidates
        for d = DelayCandidates
            % WD uses Hard Decisions (Teacher Forcing with True Tx during probe)
            [X, Y] = build_data(Rx, Tx_n, InputLength, k, off, d, ScanSamples);
            if size(X,2)<500, continue; end
            X=double(X.'); Y=double(Y.');
            R = X'*X; w = (R + 1e-3*eye(size(R))) \ (X'*Y);
            mse = mean((X*w - Y).^2);
            if mse < best_mse, best_mse=mse; best_delay=d; best_offset=off; w_lin=w; end
        end
    end

    %% 3) Train NN (Residuals)
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

    %% 4) Inference (Feedback Loop)
    [Rx_Mat, ~, all_idx] = build_rx_only(Rx, Tx_n, InputLength, best_offset, best_delay);
    N_tot = size(Rx_Mat, 2);
    ye_tot = zeros(N_tot, 1);
    
    % Get PAM levels for Hard Decision
    [~, C] = kmeans(double(Y_tr(1:min(2000,end))), 4, 'Replicates',3);
    lvls = sort(C).'; thr = (lvls(1:3)+lvls(2:4))/2;
    
    % Extract Weights
    W1=gather(net.Layers(2).Weights); b1=gather(net.Layers(2).Bias);
    W2=gather(net.Layers(4).Weights); b2=gather(net.Layers(4).Bias);
    W3=gather(net.Layers(6).Weights); b3=gather(net.Layers(6).Bias);
    
    fb = zeros(k,1); % Init buffer
    
    for n = 1:N_tot
        in_vec = [Rx_Mat(:,n); fb];
        
        % Linear + Residual
        y_lin = w_lin.' * double(in_vec);
        h1 = tanh(W1*in_vec + b1);
        h2 = tanh(W2*h1 + b2);
        y_res = W3*h2 + b3;
        
        y_out = y_lin + y_res;
        ye_tot(n) = y_out;
        
        % Hard Decision for Feedback
        y_hard = hard_dec(y_out, lvls, thr);
        fb = [y_hard; fb(1:end-1)];
    end

    ye = zeros(length(Tx),1); ye(all_idx) = ye_tot * y_std + y_mean;
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
        fb_p = Tx(n(i)-1 : -1 : n(i)-k); % True Tx as Hard Decision proxy
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

function yq = hard_dec(y, L, T)
    if y<T(1), yq=L(1); elseif y<T(2), yq=L(2); elseif y<T(3), yq=L(3); else, yq=L(4); end
end

function f = canUseGPU_local(), try, d=gpuDevice; f=d.Index>0; catch, f=false; end, end
