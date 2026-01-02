function [ye, net, valid_tx_indices, best_delay, best_offset] = RNN_WD_Implementation( ...
    xRx, xTx, NumPreamble_TDE, InputLength, HiddenSize, LearningRate, MaxEpochs, k, ...
    DelayCandidates, OffsetCandidates, NoiseStd)
% RNN_WD_Implementation (Residual Learning Version)
%   Structure: Output = Linear_FFE(Rx) + RNN_Residual(Rx, Feedback)
%   Input: [Rx_Window; Hard_Decision(n-1...n-k)]
%   Output: Estimated Symbol Level y(n) (Normalized)

    %% 0) Defaults
    if nargin < 4 || isempty(InputLength), InputLength = 21; end
    if nargin < 5 || isempty(HiddenSize), HiddenSize = 32; end
    if nargin < 6 || isempty(LearningRate), LearningRate = 1e-3; end
    if nargin < 7 || isempty(MaxEpochs), MaxEpochs = 30; end
    if nargin < 8 || isempty(k), k = 5; end 
    if nargin < 9 || isempty(DelayCandidates), DelayCandidates = -10:10; end
    if nargin < 10 || isempty(OffsetCandidates), OffsetCandidates = [1 2]; end
    if nargin < 11 || isempty(NoiseStd), NoiseStd = 0.2; end

    execEnv = 'auto';
    try, if canUseGPU_local() && exist('gpuArray','file')==2, execEnv = 'gpu'; end; catch, execEnv='cpu'; end

    %% 1) Preprocess
    Rx = xRx(:); Tx = xTx(:);
    
    y_mean = mean(Tx); y_std  = std(Tx);
    Rx = (Rx - mean(Rx)) / std(Rx);
    Tx_n = (Tx - y_mean) / y_std; 

    %% 2) Linear Probe (FFE Baseline) & Alignment
    ScanSamples = min(4000, NumPreamble_TDE);
    best_mse = inf; best_delay = 0; best_offset = OffsetCandidates(1); 
    w_lin = zeros(InputLength, 1);
    
    for off = OffsetCandidates
        for d = DelayCandidates
            % Linear probe only uses Rx, no feedback (k=0)
            [X, Y] = build_data(Rx, Tx_n, InputLength, 0, off, d, ScanSamples, 0); 
            if size(X,2)<500, continue; end
            X=double(X.'); Y=double(Y.');
            % Ridge Regression
            w = (X'*X + 1e-3*eye(size(X,2))) \ (X'*Y);
            mse = mean((X*w - Y).^2);
            if mse < best_mse, best_mse=mse; best_delay=d; best_offset=off; w_lin=w; end
        end
    end

    %% 3) Train NN (Residuals)
    % Prepare data with Feedback (Teacher Forcing with Noise)
    [X_all, Y_all, valid_tx_indices] = build_data(Rx, Tx_n, InputLength, k, best_offset, best_delay, NumPreamble_TDE, NoiseStd);
    X_tr = X_all.'; 
    Y_tr = Y_all.';
    
    % Calculate Linear Prediction (Part 1)
    % Note: X_tr contains [Rx; Feedback]. w_lin only applies to Rx part.
    X_rx_tr = double(X_tr(:, 1:InputLength)); 
    Y_lin_tr = X_rx_tr * w_lin;
    
    % Calculate Residual Target
    Y_res_tr = Y_tr - single(Y_lin_tr);
    
    layers = [
        featureInputLayer(InputLength+k, 'Normalization','none')
        fullyConnectedLayer(HiddenSize, 'WeightsInitializer','he')
        tanhLayer % Tanh is good for residuals centered at 0
        fullyConnectedLayer(ceil(HiddenSize/2), 'WeightsInitializer','he')
        tanhLayer
        fullyConnectedLayer(1)
        regressionLayer
    ];
    
    opts = trainingOptions('adam', 'MaxEpochs', MaxEpochs, 'MiniBatchSize', 256, ...
        'InitialLearnRate', LearningRate, 'L2Regularization', 1e-4, ...
        'Shuffle','every-epoch', 'Verbose',0, 'Plots','none', 'ExecutionEnvironment', execEnv);
    
    net = trainNetwork(X_tr, Y_res_tr, layers, opts);

    % --- DEBUG INFO START ---
    Y_pred_res = predict(net, X_tr, 'ExecutionEnvironment', execEnv);
    if isa(Y_pred_res, 'gpuArray'), Y_pred_res = gather(Y_pred_res); end
    
    % Final prediction on training set = Linear + RNN_Residual
    Y_final_tr = Y_lin_tr + double(Y_pred_res);
    
    mse_lin = mean((Y_tr - Y_lin_tr).^2);
    mse_final = mean((Y_tr - Y_final_tr).^2);
    
    % Hard Decision for Training BER
    [~, C_tr] = kmeans(double(Y_tr(1:min(5000,end))), 4, 'Replicates',3);
    lvls_tr = sort(C_tr); thr_tr = (lvls_tr(1:3)+lvls_tr(2:4))/2;
    
    y_hard_tr = hard_dec_vec(Y_final_tr, lvls_tr, thr_tr);
    tx_hard_tr = hard_dec_vec(Y_tr, lvls_tr, thr_tr);
    
    [~, ber_train] = biterr(y_hard_tr, tx_hard_tr, 2);
    
    disp('---------------------------------------------------------');
    disp(['[RNN_WD Residual] Linear MSE: ', num2str(mse_lin, '%.5e')]);
    disp(['[RNN_WD Residual] Final MSE:  ', num2str(mse_final, '%.5e')]);
    disp(['[RNN_WD Residual] Train BER:  ', num2str(ber_train, '%.5e')]);
    disp('---------------------------------------------------------');
    % --- DEBUG INFO END ---

    %% 4) Inference (Hard Decision Feedback Loop)
    [Rx_Mat, ~, all_idx] = build_rx_only(Rx, Tx_n, InputLength, best_offset, best_delay);
    
    N_tot = size(Rx_Mat, 2);
    ye_tot = zeros(N_tot, 1);
    
    % Determine Levels for Hard Decision
    [~, C] = kmeans(double(Y_tr(1:min(2000,end))), 4, 'Replicates',3);
    lvls = sort(C).'; thr = (lvls(1:3)+lvls(2:4))/2;

    % Extract Weights
    W1=gather(net.Layers(2).Weights); b1=gather(net.Layers(2).Bias);
    W2=gather(net.Layers(4).Weights); b2=gather(net.Layers(4).Bias);
    W3=gather(net.Layers(6).Weights); b3=gather(net.Layers(6).Bias);
    
    fb = zeros(k,1); 
    
    for n = 1:N_tot
        % 1. Linear Part
        rx_vec = double(Rx_Mat(:,n));
        y_lin = w_lin.' * rx_vec;
        
        % 2. RNN Part
        in_vec = [Rx_Mat(:,n); fb];
        h1 = tanh(W1*in_vec + b1);
        h2 = tanh(W2*h1 + b2);
        y_res = W3*h2 + b3;
        
        % 3. Combine
        y_out = y_lin + y_res;
        ye_tot(n) = y_out;
        
        % Hard Decision Feedback
        y_hard = hard_dec(y_out, lvls, thr);
        fb = [y_hard; fb(1:end-1)];
    end

    ye = zeros(length(Tx),1); 
    ye(all_idx) = ye_tot * y_std + y_mean;
    valid_tx_indices = all_idx;
end

function [X, Y, idx] = build_data(Rx, Tx, L, k, off, d, max_n, noise_std)
    if nargin < 8, noise_std = 0; end
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
        
        if k > 0
            fb_p = Tx(n(i)-1 : -1 : n(i)-k); 
            if noise_std > 0
                fb_p = fb_p + noise_std * randn(size(fb_p));
            end
        else
            fb_p = [];
        end
        
        X(:,i) = [rx_p; fb_p];
    end
    idx = n;
    Y = Y.';
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

function yq = hard_dec_vec(y, L, T)
    yq = zeros(size(y));
    yq(y < T(1)) = 0;
    yq(y >= T(1) & y < T(2)) = 1;
    yq(y >= T(2) & y < T(3)) = 2;
    yq(y >= T(3)) = 3;
end

function f = canUseGPU_local(), try, d=gpuDevice; f=d.Index>0; catch, f=false; end, end