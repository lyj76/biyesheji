function [ye, net, valid_tx_indices, best_delay, best_offset] = WDRNN_Implementation( ...
    xRx, xTx, NumPreamble_TDE, InputLength, HiddenSize, LearningRate, MaxEpochs, k, ...
    DelayCandidates, OffsetCandidates)
% WDRNN_Implementation - Weighted Decision RNN (Paper Replication)
% Single Hidden Layer + Weighted Feedback
% Robust version aligned with RNN_Implementation structure

    %% ===== defaults =====
    if nargin < 4 || isempty(InputLength), InputLength = 61; end
    if nargin < 5 || isempty(HiddenSize), HiddenSize = 20; end
    if nargin < 6 || isempty(LearningRate), LearningRate = 1e-3; end
    if nargin < 7 || isempty(MaxEpochs), MaxEpochs = 50; end
    if nargin < 8 || isempty(k), k = 25; end
    if nargin < 9 || isempty(DelayCandidates), DelayCandidates = -20:20; end
    if nargin < 10 || isempty(OffsetCandidates), OffsetCandidates = [1 2]; end
    
    ScanTrainSamples = min(5000, NumPreamble_TDE);
    ScanValSamples = min(2000, max(0, NumPreamble_TDE - ScanTrainSamples));

    if mod(InputLength,2)==0
        warning('[WDRNN] InputLength even; recommend odd for centered window.');
    end

    %% ===== WD Parameters (Paper) =====
    alpha = 5;
    beta = 0.14;

    %% ===== Preprocess & Normalize =====
    Rx = xRx(:);
    Tx = xTx(:);

    y_mean = mean(Tx);
    y_std  = std(Tx);

    % Normalize to ~N(0,1) for training stability
    Rx = (Rx - mean(Rx)) / std(Rx);
    Tx_n = (Tx - y_mean) / y_std;

    %% ===== 1) Sync: Scan offset/delay (Linear Probe) =====
    best_ser = inf;
    best_mse = inf;
    best_delay = DelayCandidates(1);
    best_offset = OffsetCandidates(1);

    % Learn levels from normalized Tx for robust sync check
    Yref = double(Tx_n(1:min(NumPreamble_TDE, length(Tx_n))));
    [~, Cref] = kmeans(Yref(:), 4, 'Replicates', 3);
    levels_ref = sort(Cref(:)).';
    thr_ref = (levels_ref(1:3) + levels_ref(2:4))/2;

    for oi = 1:numel(OffsetCandidates)
        offset = OffsetCandidates(oi);
        for di = 1:numel(DelayCandidates)
            delay = DelayCandidates(di);
            
            [Xscan, Yscan] = build_center_window_dataset(Rx, Tx_n, InputLength, offset, delay, ScanTrainSamples + ScanValSamples);
            if size(Xscan,2) < (ScanTrainSamples + 10), continue; end

            Xs = Xscan.';
            Ys = Yscan.';
            
            Xtr = Xs(1:ScanTrainSamples,:);
            Ytr = Ys(1:ScanTrainSamples,:);
            
            if size(Xtr,1) <= size(Xtr,2), continue; end

            w = Xtr \ Ytr; % Linear probe
            Yp = Xtr * w;
            
            % Check SER
            Yp_q = hard_slice_pam4(Yp, levels_ref, thr_ref);
            Yv_q = hard_slice_pam4(Ytr, levels_ref, thr_ref);
            ser = mean(Yp_q ~= Yv_q);

            if ser < best_ser
                best_ser = ser;
                best_mse = mean((Yp - Ytr).^2);
                best_delay = delay;
                best_offset = offset;
            end
        end
    end
    
    % disp(['    [WDRNN] Sync: Delay=', num2str(best_delay), ', Offset=', num2str(best_offset), ', SER=', num2str(best_ser)]);

    %% ===== 2) Build Full Aligned Dataset =====
    [Xall, Yall, valid_tx_indices] = build_center_window_dataset(Rx, Tx_n, InputLength, best_offset, best_delay, []);
    Nvalid = size(Xall,2);
    Ntrain = min(NumPreamble_TDE, Nvalid);

    %% ===== 3) Training Set (Teacher Forcing) =====
    start_n = k + 1;
    end_n   = Ntrain;

    Xar = zeros(InputLength + k, end_n - start_n + 1, 'single');
    Yar = zeros(1, end_n - start_n + 1, 'single');

    idx = 1;
    for n = start_n:end_n
        fb = Yall(1, n-1 : -1 : n-k);       % True feedback
        Xar(:,idx) = [Xall(:,n); fb.'];     
        Yar(1,idx) = Yall(1,n);
        idx = idx + 1;
    end
    
    X_Train = Xar.';   
    Y_Train = Yar.';   

    %% ===== 4) Network (Single Hidden Layer) =====
    layers = [
        featureInputLayer(InputLength + k, 'Normalization','none', 'Name','input')
        fullyConnectedLayer(HiddenSize, 'Name','fc1', 'WeightsInitializer', 'he')
        tanhLayer('Name','tanh1')
        % No Dropout for strict paper replication
        fullyConnectedLayer(1, 'Name','out', 'WeightsInitializer', 'he')
        regressionLayer('Name','loss')
    ];

    options = trainingOptions('adam', ...
        'MaxEpochs', MaxEpochs, ...
        'MiniBatchSize', 256, ...
        'InitialLearnRate', LearningRate, ...
        'L2Regularization', 1e-4, ...
        'Shuffle','every-epoch', ...
        'Verbose', 0, ...
        'ExecutionEnvironment', 'auto');

    %% ===== 5) Train =====
    net = trainNetwork(X_Train, Y_Train, layers, options);

    %% ===== 6) Inference: Weighted Feedback (WD) =====
    % Extract Weights
    L = net.Layers;
    l_fc1 = L(strcmp({L.Name}, 'fc1'));
    l_out = L(strcmp({L.Name}, 'out'));

    W1 = gather(l_fc1.Weights); b1 = gather(l_fc1.Bias);
    W2 = gather(l_out.Weights); b2 = gather(l_out.Bias);

    % Learn levels from training data for WD calculation
    % (Network operates in normalized domain, so levels must be normalized)
    Ytr_vals = double(Yall(1, 1:Ntrain)).';
    [~, C] = kmeans(Ytr_vals, 4, 'Replicates', 3);
    pam4_levels = sort(C(:)).'; 
    thr = (pam4_levels(1:3) + pam4_levels(2:4))/2;
    
    % Prepare buffers
    tanhf = @(z) tanh(z);
    ye_n  = zeros(Nvalid,1);   
    ye_fb = zeros(Nvalid,1); % This stores the FEEDBACK value (Weighted)

    % Init Preamble (Teacher Forcing for first k)
    ye_n(1:k)  = Yall(1,1:k).';
    ye_fb(1:k) = Yall(1,1:k).'; % Or hard sliced, but truth is better for start

    for n = (k+1):Nvalid
        % Feedback Vector (Weighted history)
        fb = ye_fb(n-1:-1:n-k); 
        
        u  = [Xall(:,n); single(fb)];
        u  = double(u);

        % Forward
        h1 = tanhf(W1*u + b1);
        y_soft = W2*h1 + b2;  % Soft Output
        ye_n(n) = y_soft;

        % --- WD Logic ---
        % 1. Hard Decision
        y_hard = hard_slice_pam4(y_soft, pam4_levels, thr);
        
        % 2. Reliability Gamma
        % Note: y_soft and pam4_levels are normalized. 
        % Paper gamma assumes levels are -3,-1,1,3. We need to scale distance?
        % Actually paper formula: gamma = 1 - |y - d|. If distance > 1, gamma < 0?
        % Paper says gamma clipped [0,1].
        % Let's use relative distance normalized by min distance between levels.
        % Avg distance between levels in norm domain:
        dist_levels = mean(diff(pam4_levels));
        
        abs_err = abs(y_soft - y_hard);
        % Normalize error to be comparable to paper's scale (where dist=2)
        % Paper: levels -3,-1 -> dist 2. gamma = 1 - |e|. |e|<1 means reliable.
        % Here: dist = dist_levels. We map |e|/dist_levels to paper's |e|/2
        % Simplified: gamma = 1 - abs(y_soft - y_hard) / (dist_levels/2);
        
        % Let's stick to paper formula strictly but scale the input to -3..3 first?
        % Better: scale the output y_soft back to -3..3 logic just for WD calc.
        % Mapping: (y - mean) / std -> y_original.
        % But we don't have perfect -3..3 recovery yet.
        
        % Alternative: Adaptive WD
        % gamma = 1 - |y - d| / (half_symbol_spacing)
        half_dist = dist_levels / 2;
        gamma_n = 1 - abs_err / half_dist; 
        if gamma_n < 0, gamma_n = 0; end
        
        % 3. Weight S
        term = -alpha * (gamma_n/beta - 1);
        S_val = 0.5 * ( (1 - exp(term)) / (1 + exp(term)) + 1 );
        
        % 4. Weighted Feedback
        y_fb_val = S_val * y_hard + (1 - S_val) * y_soft;
        ye_fb(n) = y_fb_val;
    end
    
    % De-normalize for final output
    ye_val = ye_n * y_std + y_mean;
    ye = zeros(length(Tx),1);
    ye(valid_tx_indices) = ye_val;

end

%% ================= helper functions =================
function [X, Y, tx_idx_out] = build_center_window_dataset(Rx_Data, Tx_Data, InputLength, offset, delay, max_samples)
    HalfLen = floor(InputLength/2);
    max_sym_rx = floor((length(Rx_Data) - 1 - offset)/2) + 1;
    if max_sym_rx < 1, X=[]; Y=[]; tx_idx_out=[]; return; end

    sym_idx = (1:max_sym_rx).';
    tx_idx  = sym_idx + delay;

    valid_mask = (tx_idx >= 1) & (tx_idx <= length(Tx_Data));
    sym_idx = sym_idx(valid_mask);
    tx_idx  = tx_idx(valid_mask);

    if isempty(sym_idx), X=[]; Y=[]; tx_idx_out=[]; return; end

    center = (sym_idx - 1) * 2 + offset;
    start_i = center - HalfLen;
    end_i   = center + HalfLen;

    valid2 = (start_i >= 1) & (end_i <= length(Rx_Data));
    sym_idx = sym_idx(valid2);
    tx_idx  = tx_idx(valid2);
    center  = center(valid2);

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

function yq = hard_slice_pam4(y, levels, thr)
    y = double(y);
    yq = zeros(size(y));
    yq(y < thr(1)) = levels(1);
    yq(y >= thr(1) & y < thr(2)) = levels(2);
    yq(y >= thr(2) & y < thr(3)) = levels(3);
    yq(y >= thr(3)) = levels(4);
end