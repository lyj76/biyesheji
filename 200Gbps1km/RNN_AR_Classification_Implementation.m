function [ye_valid, ysym_valid, valid_idx, net, info] = RNN_AR_Classification_Implementation( ...
    xRx, xTx, NumPreamble_TDE, InputLength, HiddenSize, LearningRate, MaxEpochs, k, ...
    DelayCandidates, OffsetCandidates, ...
    ScanTrainSamples, ScanValSamples, ...
    WarmupSymbols, SS_StartProb, SS_EndProb)
% RNN_AR_Classification_Implementation
% - AR-MLP classifier: input = centered window (Rx) + k past decisions (one-hot)
% - Training: teacher forcing + scheduled sampling (custom loop)
% - Inference: warm-up then free-running
%
% Outputs:
%   ye_valid: amplitude-domain mapped PAM levels
%   ysym_valid: 0..M-1 class indices
%   valid_idx: Tx-domain indices aligned
%   net: dlnetwork
%   info: struct with best_offset/best_delay and pam levels

    %% defaults
    if nargin < 4 || isempty(InputLength), InputLength = 41; end
    if nargin < 5 || isempty(HiddenSize), HiddenSize = 64; end
    if nargin < 6 || isempty(LearningRate), LearningRate = 1e-3; end
    if nargin < 7 || isempty(MaxEpochs), MaxEpochs = 15; end
    if nargin < 8 || isempty(k), k = 2; end
    if nargin < 9 || isempty(DelayCandidates), DelayCandidates = -30:30; end
    if nargin < 10 || isempty(OffsetCandidates), OffsetCandidates = [1 2]; end
    if nargin < 11 || isempty(ScanTrainSamples), ScanTrainSamples = min(5000, NumPreamble_TDE); end
    if nargin < 12 || isempty(ScanValSamples), ScanValSamples = min(2000, max(0, NumPreamble_TDE-ScanTrainSamples)); end
    if nargin < 13 || isempty(WarmupSymbols), WarmupSymbols = 200; end
    if nargin < 14 || isempty(SS_StartProb), SS_StartProb = 0.0; end
    if nargin < 15 || isempty(SS_EndProb), SS_EndProb = 0.3; end

    M = 4;

    execEnv = "cpu";
    useGPU = false;
    try
        d = gpuDevice;
        if d.DeviceAvailable
            execEnv = "gpu";
            useGPU = true;
        end
    catch
    end

    Rx = xRx(:);
    Tx = xTx(:);
    Rx = (Rx - mean(Rx)) / std(Rx);

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

    fprintf('    [AR-CLS] Best Offset=%d, Best Delay=%d, scan MSE=%.5g\n', best_offset, best_delay, best_mse);

    %% build aligned full windows
    [Xall, ~, valid_idx] = build_center_window_dataset_2sps(Rx, Tx, InputLength, best_offset, best_delay, []);
    if isempty(valid_idx)
        error('[AR-CLS] No valid samples. Check offset/delay/window.');
    end
    yall_sym = tx_sym(valid_idx); % 0..3 aligned

    Nvalid = numel(valid_idx);
    Ntrain = min(NumPreamble_TDE, Nvalid);

    t0 = k + 1;
    t1 = Ntrain;
    if t1 <= t0
        error('[AR-CLS] Not enough training samples for k=%d.', k);
    end

    Yref = double(Tx(1:min(NumPreamble_TDE, numel(Tx))));
    [~, C] = kmeans(Yref(:), M, 'Replicates', 5);
    pam_levels = sort(C(:));

    %% dlnetwork
    F = InputLength + k*M;

    lgraph = layerGraph([
        featureInputLayer(F, 'Normalization','none', 'Name','in')
        fullyConnectedLayer(HiddenSize, 'Name','fc1')
        tanhLayer('Name','tanh1')
        fullyConnectedLayer(64, 'Name','fc2')
        tanhLayer('Name','tanh2')
        fullyConnectedLayer(M, 'Name','fc_out')
    ]);

    net = dlnetwork(lgraph);

    %% custom training loop with scheduled sampling
    mb = 512;
    numItersPerEpoch = ceil((t1 - t0 + 1)/mb);

    trailingAvg = [];
    trailingAvgSq = [];
    iteration = 0;

    fprintf('    [AR-CLS] Training (Samples=%d, Epochs=%d, SS %.2f->%.2f)...\n', (t1-t0+1), MaxEpochs, SS_StartProb, SS_EndProb);

    Xall_s = single(Xall);
    ytrue  = yall_sym(:);

    for epoch = 1:MaxEpochs
        if MaxEpochs == 1
            ssProb = SS_EndProb;
        else
            ssProb = SS_StartProb + (SS_EndProb-SS_StartProb)*(epoch-1)/(MaxEpochs-1);
        end

        idx_epoch = (t0:t1).';
        idx_epoch = idx_epoch(randperm(numel(idx_epoch)));

        for it = 1:numItersPerEpoch
            iteration = iteration + 1;

            i1 = (it-1)*mb + 1;
            i2 = min(it*mb, numel(idx_epoch));
            bt = idx_epoch(i1:i2);

            Xb = zeros(F, numel(bt), 'single');
            Tb = zeros(M, numel(bt), 'single');

            for j = 1:numel(bt)
                n = bt(j);

                xw = Xall_s(:, n);

                hist_true = ytrue(n-1:-1:n-k);
                fb_true = onehot_hist(hist_true, M);

                fb_pred = fb_true;
                if ssProb > 0
                    mask = rand(k,1) < ssProb;
                    for kk = 1:k
                        if mask(kk)
                            c = hist_true(kk);
                            c2 = max(0, min(M-1, c + randi([-1 1])));
                            fb_pred((kk-1)*M+1:kk*M) = 0;
                            fb_pred((kk-1)*M + (c2+1)) = 1;
                        end
                    end
                end

                Xb(:,j) = [xw; fb_pred];

                cT = ytrue(n);
                Tb(:,j) = 0;
                Tb(cT+1, j) = 1;
            end

            dlX = dlarray(Xb, 'CB');
            dlT = dlarray(Tb, 'CB');

            if useGPU
                dlX = gpuArray(dlX);
                dlT = gpuArray(dlT);
            end

            [loss, grads] = dlfeval(@modelGradients, net, dlX, dlT);
            [net, trailingAvg, trailingAvgSq] = adamupdate(net, grads, trailingAvg, trailingAvgSq, iteration, LearningRate);
        end

        if mod(epoch, 5) == 0 || epoch == 1 || epoch == MaxEpochs
            fprintf('    [AR-CLS] Epoch %d/%d done. ssProb=%.3f\n', epoch, MaxEpochs, ssProb);
        end
    end

    %% inference: warm-up then free-running
    fprintf('    [AR-CLS] Inference (warm-up=%d symbols) ...\n', WarmupSymbols);

    ysym_valid = zeros(Nvalid,1,'int32');

    ysym_valid(1:k) = int32(ytrue(1:k));

    for n = (k+1):Nvalid
        if n <= WarmupSymbols
            hist = ytrue(n-1:-1:n-k);
        else
            hist = double(ysym_valid(n-1:-1:n-k));
        end

        fb = onehot_hist(hist, M);
        xw = Xall_s(:, n);
        u  = [xw; fb];

        dlU = dlarray(u, 'CB');
        if useGPU, dlU = gpuArray(dlU); end

        logits = forward(net, dlU);
        [~, cls] = max(extractdata(gather(logits)), [], 1);
        ysym_valid(n) = int32(cls - 1);
    end

    ye_valid = pam_levels(double(ysym_valid)+1);

    %% info
    info.best_offset = best_offset;
    info.best_delay  = best_delay;
    info.scan_mse    = best_mse;
    info.pam_levels  = pam_levels(:).';
    info.warmup      = WarmupSymbols;
    info.ss_start    = SS_StartProb;
    info.ss_end      = SS_EndProb;
    info.k           = k;
end

function [loss, grads] = modelGradients(net, dlX, dlT)
    logits = forward(net, dlX);
    dlP = softmax(logits);
    epsv = 1e-8;
    loss = -mean(sum(dlT .* log(dlP + epsv), 1), 2);
    grads = dlgradient(loss, net.Learnables);
end

function fb = onehot_hist(hist_classes, M)
    k = numel(hist_classes);
    fb = zeros(k*M, 1, 'single');
    for i = 1:k
        c = hist_classes(i);
        fb((i-1)*M + (c+1)) = 1;
    end
end

function [X, Y, idxTx] = build_center_window_dataset_2sps(Rx, Tx, L, offset, delay, max_samples)
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
