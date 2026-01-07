%% Test NN baselines (MATLAB trainNetwork/predict)
clear;
close all;

%% add paths
addpath('fns');
addpath(fullfile('fns','fns2'));

%% parameters
Ft = 200e9;
Osamp_factor = 2;
NumSymbols = 2^18;
M = 4;

%% PAM4 modulate
[xsym, xm] = PAMSource(M, NumSymbols);
xsym = xsym(:);
xm = xm(:);
xs = xm;

%% pulse shaping
rolloff = 0.1;
N = 128;
h = fdesign.pulseshaping(Osamp_factor, 'Raised Cosine', 'N,Beta', N, rolloff);
Hd = design(h);
sqrt_ht = Hd.Numerator;
sqrt_ht = sqrt_ht ./ max(sqrt_ht);

x_upsamp = upsample(xs, Osamp_factor);
x_shape = conv(sqrt_ht, x_upsamp);
x_shape = x_shape ./ sqrt(mean(abs(x_shape).^2));

%% data list
file_list = {'rop3dBm_1.mat', 'rop5dBm_1.mat'};

%% common settings
NumPreamble_TDE = 10000;

%% FNN baseline settings
FNN_L = 101;
FNN_H1 = 128;
FNN_H2 = 64;
FNN_LR = 1e-3;
FNN_Epochs = 30;
FNN_DelayCandidates = -30:30;
FNN_OffsetCandidates = [1 2];
FNN_ScanTrain = 5000;
FNN_ScanVal = 2000;

%% RNN baseline settings (GRU)
RNN_L = 41;
RNN_H = 64;
RNN_LR = 1e-3;
RNN_Epochs = 15;
RNN_DelayCandidates = -30:30;
RNN_OffsetCandidates = [1 2];
RNN_ScanTrain = 5000;
RNN_ScanVal = 2000;

for n1 = 1:length(file_list)
    disp(['Processing File Index: ', num2str(n1)]);
    load(file_list{n1}, 'ReData');

    ReData = -ReData;

    %% synchronization
    th = 0.3;
    [TE, FE] = TEFEMMM2(ReData, 1024, th);
    TE = TE + 0;

    ysync = ReData(1024 + 20 + TE : end);
    ysync = ysync(1 : length(x_shape));

    %% match filtering
    yt_filter = ysync(1 + N/2 : length(ysync) - N/2);

    %% equalizer inputs
    xTx = xs;
    xRx = yt_filter;

    %% FNN baseline (regression)
    [best_off, best_delay] = scan_best_offset_delay(xRx, xTx, FNN_L, FNN_DelayCandidates, FNN_OffsetCandidates, FNN_ScanTrain, FNN_ScanVal);
    disp(['[FNN] Best Offset = ', num2str(best_off), ', Best Delay = ', num2str(best_delay)]);

    [Xall, idxTx] = build_center_window_dataset_2sps(xRx, xTx, FNN_L, best_off, best_delay, []);
    idxTx = double(idxTx(:));
    if size(Xall,2) ~= numel(idxTx)
        Nvalid = min(size(Xall,2), numel(idxTx));
        Xall = Xall(:,1:Nvalid);
        idxTx = idxTx(1:Nvalid);
    end
    mask = idxTx >= 1 & idxTx <= length(xTx) & isfinite(idxTx);
    Xall = Xall(:,mask);
    idxTx = idxTx(mask);
    Nvalid = size(Xall,2);
    Ntrain = min(NumPreamble_TDE, Nvalid);
    if Ntrain < 1
        error('[FNN] No valid training samples after alignment. Check offset/delay.');
    end

    Xtrain = Xall(:,1:Ntrain).';
    Ytrain = xTx(idxTx(1:Ntrain));

    layersF = [
        featureInputLayer(FNN_L, 'Normalization','none', 'Name','in')
        fullyConnectedLayer(FNN_H1, 'Name','fc1')
        reluLayer('Name','relu1')
        fullyConnectedLayer(FNN_H2, 'Name','fc2')
        reluLayer('Name','relu2')
        fullyConnectedLayer(1, 'Name','out')
        regressionLayer('Name','loss')
    ];

    optionsF = trainingOptions('adam', ...
        'InitialLearnRate', FNN_LR, ...
        'MaxEpochs', FNN_Epochs, ...
        'MiniBatchSize', 1024, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', 0, ...
        'Plots', 'none', ...
        'ExecutionEnvironment', 'auto');

    disp('[FNN] Training baseline...');
    netF = trainNetwork(single(Xtrain), single(Ytrain), layersF, optionsF);

    YeF = predict(netF, single(Xall.'), 'MiniBatchSize', 4096, 'ExecutionEnvironment', 'auto');
    ye_valid = double(YeF(:));

    statsF = eval_equalizer_pam4(ye_valid, idxTx, xsym, xm, NumPreamble_TDE, M);
    disp(['File ', num2str(n1), ' FNN BER = ', num2str(statsF.BER), ', SNR(dB) = ', num2str(statsF.SNRdB)]);

    %% RNN baseline (GRU, sequence-to-one regression)
    [best_off, best_delay] = scan_best_offset_delay(xRx, xTx, RNN_L, RNN_DelayCandidates, RNN_OffsetCandidates, RNN_ScanTrain, RNN_ScanVal);
    disp(['[RNN] Best Offset = ', num2str(best_off), ', Best Delay = ', num2str(best_delay)]);

    [XallR, idxTxR] = build_center_window_dataset_2sps(xRx, xTx, RNN_L, best_off, best_delay, []);
    idxTxR = double(idxTxR(:));
    if size(XallR,2) ~= numel(idxTxR)
        NvalidR = min(size(XallR,2), numel(idxTxR));
        XallR = XallR(:,1:NvalidR);
        idxTxR = idxTxR(1:NvalidR);
    end
    maskR = idxTxR >= 1 & idxTxR <= length(xTx) & isfinite(idxTxR);
    XallR = XallR(:,maskR);
    idxTxR = idxTxR(maskR);
    NvalidR = size(XallR,2);
    NtrainR = min(NumPreamble_TDE, NvalidR);
    if NtrainR < 1
        error('[RNN] No valid training samples after alignment. Check offset/delay.');
    end

    Xseq_all = make_seq_cell(XallR);
    Xseq_train = Xseq_all(1:NtrainR);
    Yseq_train = xTx(idxTxR(1:NtrainR));

    layersR = [
        sequenceInputLayer(1, 'Name','in')
        gruLayer(RNN_H, 'OutputMode','last', 'Name','gru')
        fullyConnectedLayer(1, 'Name','out')
        regressionLayer('Name','loss')
    ];

    optionsR = trainingOptions('adam', ...
        'InitialLearnRate', RNN_LR, ...
        'MaxEpochs', RNN_Epochs, ...
        'MiniBatchSize', 256, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', 0, ...
        'Plots', 'none', ...
        'ExecutionEnvironment', 'auto');

    disp('[RNN] Training baseline GRU...');
    netR = trainNetwork(Xseq_train, single(Yseq_train), layersR, optionsR);

    YeR = predict(netR, Xseq_all, 'MiniBatchSize', 512, 'ExecutionEnvironment', 'auto');
    ye_valid = double(YeR(:));

    statsR = eval_equalizer_pam4(ye_valid, idxTxR, xsym, xm, NumPreamble_TDE, M);
    disp(['File ', num2str(n1), ' RNN(GRU) BER = ', num2str(statsR.BER), ', SNR(dB) = ', num2str(statsR.SNRdB)]);
end

%% ---------------- local functions ----------------
function [best_off, best_delay] = scan_best_offset_delay(xRx, xTx, L, delay_list, offset_list, ntrain, nval)
    best_mse = inf;
    best_off = offset_list(1);
    best_delay = delay_list(1);

    for off = offset_list(:).'
        for delay = delay_list(:).'
            [X, Y, idx] = build_center_window_dataset_2sps(xRx, xTx, L, off, delay, ntrain + nval);
            if isempty(idx) || size(X,2) < (ntrain + max(1,nval))
                continue;
            end
            Xs = double(X(:,1:ntrain)).';
            Ys = double(Y(1:ntrain));
            if size(Xs,1) <= size(Xs,2)
                continue;
            end
            w = Xs \ Ys;
            if nval > 0
                Xv = double(X(:,ntrain+1:ntrain+nval)).';
                Yv = double(Y(ntrain+1:ntrain+nval));
                Yp = Xv * w;
                mse = mean((Yp - Yv).^2);
            else
                Yp = Xs * w;
                mse = mean((Yp - Ys).^2);
            end
            if mse < best_mse
                best_mse = mse;
                best_off = off;
                best_delay = delay;
            end
        end
    end
end

function [X, Y, idxTx] = build_center_window_dataset_2sps(Rx, Tx, L, offset, delay, max_samples)
    Rx = Rx(:);
    Tx = Tx(:);
    HalfLen = floor(L/2);

    max_sym_rx = floor((length(Rx) - offset)/2) + 1;
    if max_sym_rx <= 0
        X = []; Y = []; idxTx = [];
        return;
    end

    sym_rx = (1:max_sym_rx).';
    idxTx = sym_rx + delay;

    valid = idxTx >= 1 & idxTx <= length(Tx);
    sym_rx = sym_rx(valid);
    idxTx = idxTx(valid);

    if isempty(sym_rx)
        X = []; Y = []; idxTx = [];
        return;
    end

    center = (sym_rx - 1) * 2 + offset;
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

function Xseq = make_seq_cell(X)
    N = size(X,2);
    Xseq = cell(N,1);
    for i = 1:N
        Xseq{i,1} = single(X(:,i).'); % [1 x L]
    end
end
