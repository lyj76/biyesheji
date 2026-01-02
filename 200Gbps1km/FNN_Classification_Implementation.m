function [ye_valid, ysym_valid, valid_idx, net, info] = FNN_Classification_Implementation( ...
    xRx, xTx, NumPreamble_TDE, InputLength, HiddenSize, LearningRate, MaxEpochs, ...
    DelayCandidates, OffsetCandidates, ScanTrainSamples, ScanValSamples)
% FNN_Classification_Implementation
% - 2sps centered window features
% - softmax classification (cross entropy)
% - outputs:
%   ye_valid: amplitude domain mapped PAM levels
%   ysym_valid: class indices 0..M-1
%   valid_idx: Tx-domain indices aligned
% Notes:
% - xTx is amplitude sequence (PAM levels), NOT 0..M-1 indices
% - labels derived from xTx by pamdemod using Gray map

    %% defaults
    if nargin < 4 || isempty(InputLength), InputLength = 101; end
    if nargin < 5 || isempty(HiddenSize), HiddenSize = 64; end
    if nargin < 6 || isempty(LearningRate), LearningRate = 1e-3; end
    if nargin < 7 || isempty(MaxEpochs), MaxEpochs = 30; end
    if nargin < 8 || isempty(DelayCandidates), DelayCandidates = -30:30; end
    if nargin < 9 || isempty(OffsetCandidates), OffsetCandidates = [1 2]; end
    if nargin < 10 || isempty(ScanTrainSamples), ScanTrainSamples = min(5000, NumPreamble_TDE); end
    if nargin < 11 || isempty(ScanValSamples), ScanValSamples = min(2000, max(0, NumPreamble_TDE-ScanTrainSamples)); end

    M = 4;

    execEnv = 'cpu';
    try
        d = gpuDevice;
        if d.DeviceAvailable, execEnv = 'gpu'; end
    catch
    end

    Rx = xRx(:);
    Tx = xTx(:);

    % normalize Rx only
    Rx = (Rx - mean(Rx)) / std(Rx);

    % labels from Tx amplitude
    tx_sym = pamdemod(Tx, M, 0, 'gray'); % 0..3

    %% scan best offset/delay (linear probe on amplitude)
    best_mse = inf;
    best_offset = OffsetCandidates(1);
    best_delay  = DelayCandidates(1);

    for off = OffsetCandidates(:).'
        for delay = DelayCandidates(:).'
            [Xtmp, Ytmp, idx] = build_center_window_dataset_2sps(Rx, Tx, InputLength, off, delay, ScanTrainSamples+ScanValSamples);
            if isempty(idx) || size(Xtmp,2) < (ScanTrainSamples+max(1,ScanValSamples)), continue; end

            Xs = double(Xtmp(:,1:ScanTrainSamples)).';
            Ys = double(Ytmp(1:ScanTrainSamples));
            if size(Xs,1) <= size(Xs,2), continue; end

            w = Xs \ Ys;

            if ScanValSamples > 0
                Xv = double(Xtmp(:,ScanTrainSamples+1:ScanTrainSamples+ScanValSamples)).';
                Yv = double(Ytmp(ScanTrainSamples+1:ScanTrainSamples+ScanValSamples));
                Yp = Xv*w;
                mse = mean((Yp - Yv).^2);
            else
                Yp = Xs*w;
                mse = mean((Yp - Ys).^2);
            end

            if mse < best_mse
                best_mse = mse;
                best_offset = off;
                best_delay  = delay;
            end
        end
    end

    fprintf('    [FNN-CLS] Best Offset=%d, Best Delay=%d, scan MSE=%.5g\n', best_offset, best_delay, best_mse);

    %% build training set
    [Xall, ~, valid_idx] = build_center_window_dataset_2sps(Rx, Tx, InputLength, best_offset, best_delay, []);
    if isempty(valid_idx)
        error('[FNN-CLS] No valid samples. Check offset/delay/window.');
    end

    yall_sym = tx_sym(valid_idx) + 1;  % 1..4 for categorical

    Nvalid = numel(valid_idx);
    Ntrain = min(NumPreamble_TDE, Nvalid);

    Xtrain = Xall(:,1:Ntrain).';
    Ytrain = categorical(yall_sym(1:Ntrain));

    %% network
    layers = [
        featureInputLayer(InputLength, 'Normalization','none', 'Name','in')
        fullyConnectedLayer(HiddenSize, 'Name','fc1')
        reluLayer('Name','relu1')
        fullyConnectedLayer(64, 'Name','fc2')
        reluLayer('Name','relu2')
        fullyConnectedLayer(M, 'Name','fc_out')
        softmaxLayer('Name','sm')
        classificationLayer('Name','ce')
    ];

    options = trainingOptions('adam', ...
        'InitialLearnRate', LearningRate, ...
        'MaxEpochs', MaxEpochs, ...
        'MiniBatchSize', 1024, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', 0, ...
        'ExecutionEnvironment', execEnv, ...
        'Plots', 'none');

    fprintf('    [FNN-CLS] Training (N=%d, epochs=%d)...\n', size(Xtrain,1), MaxEpochs);
    net = trainNetwork(single(Xtrain), Ytrain, layers, options);

    %% inference on full sequence
    fprintf('    [FNN-CLS] Inference on full sequence...\n');
    scores = predict(net, single(Xall.'), 'MiniBatchSize', 4096, 'ExecutionEnvironment', execEnv); % [N x 4]
    [~, cls] = max(scores, [], 2);   % 1..4
    ysym_valid = cls - 1;            % 0..3

    %% map class to amplitude levels (learn levels from Tx preamble)
    Yref = double(Tx(1:min(NumPreamble_TDE, numel(Tx))));
    [~, C] = kmeans(Yref(:), M, 'Replicates', 5);
    pam_levels = sort(C(:));
    ye_valid = pam_levels(cls);

    %% info
    info.best_offset = best_offset;
    info.best_delay  = best_delay;
    info.scan_mse    = best_mse;
    info.pam_levels  = pam_levels(:).';
end

function [X, Y, idxTx] = build_center_window_dataset_2sps(Rx, Tx, L, offset, delay, max_samples)
% centered window in Rx(2sps) to Tx(1sps)
% Rx center index for symbol n: (n-1)*2 + offset
% Tx index: n + delay

    Rx = Rx(:); Tx = Tx(:);
    HalfLen = floor(L/2);

    max_sym_rx = floor((length(Rx) - offset)/2) + 1;
    if max_sym_rx <= 0
        X = []; Y = []; idxTx = [];
        return;
    end

    sym_rx = (1:max_sym_rx).';
    idxTx  = sym_rx + delay;

    valid = idxTx >= 1 & idxTx <= length(Tx);
    sym_rx = sym_rx(valid);
    idxTx  = idxTx(valid);

    if isempty(sym_rx)
        X = []; Y = []; idxTx = [];
        return;
    end

    center = (sym_rx - 1)*2 + offset;
    st = center - HalfLen;
    ed = center + HalfLen;

    valid2 = st >= 1 & ed <= length(Rx);
    st = st(valid2);
    idxTx = idxTx(valid2);

    if isempty(idxTx)
        X = []; Y = []; idxTx = [];
        return;
    end

    if ~isempty(max_samples)
        keep = min(max_samples, numel(idxTx));
        st = st(1:keep);
        idxTx = idxTx(1:keep);
    end

    N = numel(idxTx);
    X = zeros(L, N, 'single');
    for i = 1:N
        X(:,i) = single(Rx(st(i):st(i)+L-1));
    end
    Y = Tx(idxTx);
end
