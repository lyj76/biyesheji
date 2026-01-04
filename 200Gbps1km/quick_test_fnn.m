%% Quick Test for FNN vs FFE (3dBm & 5dBm only)
clear;
close all;

%% add paths
addpath('fns');
addpath(fullfile('fns','fns2'));
addpath('algos');

%% parameters
Ft = 200e9;
Osamp_factor = 2;
NumSymbols = 2^18;
M = 4;

s = RandStream.create('mt19937ar', 'seed', 529558);
prevStream = RandStream.setGlobalStream(s);

%% PAM4 modulate
[xsym, xm] = PAMSource(M, NumSymbols);
xsym = xsym(:);
xm = xm(:);
xs = xm;

%% pulse shaping
rolloff = 0.1;
N = 128;
sqrt_ht = rcosdesign(rolloff, N/Osamp_factor, Osamp_factor, 'normal');
sqrt_ht = sqrt_ht ./ max(sqrt_ht);

x_upsamp = upsample(xs, Osamp_factor);
x_shape = conv(sqrt_ht, x_upsamp);
x_shape = x_shape ./ sqrt(mean(abs(x_shape).^2));

%% Selected Files (Only 3dBm and 5dBm)
file_list = { ...
    fullfile('data','rop3dBm_1.mat'), ...
    fullfile('data','rop5dBm_1.mat') ...
};

%% Equalizer Parameters (Same as roll_7)
params.N1 = 61;
params.N2 = 7;
params.WL = 3;
params.D1 = 11;
params.D2 = 0;
params.WD = 1;
params.K_Lin = 18;
params.K_Vol = 90;
params.Lambda = 0.9999;
params.scale = M/2;

%% NN Parameters (Updated)
params.FNN_InputLength = 61;
params.FNN_HiddenSize = 41; % New size
params.FNN_LR = 0.001;
params.FNN_Epochs = 50;
params.FNN_DelayCandidates = -30:30;
params.FNN_OffsetCandidates = [1 2];

%% Selected Algos
algo_list = {'FFE', 'FNN'};

%% Main Loop
for n1 = 1:length(file_list)
    disp(['Processing File: ', file_list{n1}]);
    load(file_list{n1}, 'ReData')

    ReData = -ReData;

    %% synchronization
    th = 0.3;
    [TE, FE] = TEFEMMM2(ReData, 1024, th);
    
    ysync = ReData(1024 + 20 + TE : end);
    ysync = ysync(1 : length(x_shape));

    %% match filtering
    yt_filter = ysync(1 + N/2 : length(ysync) - N/2);

    xTx = xs;
    xRx = yt_filter;

    for a = 1:length(algo_list)
        algo_id = algo_list{a};

        % Determine NumPreamble
        if strcmp(algo_id, 'FNN')
            NumPreamble_TDE = 40000;
        else
            NumPreamble_TDE = 10000;
        end

        tic;
        [ye_use, idxTx] = run_equalizer(algo_id, xRx, xTx, xsym, NumPreamble_TDE, M, params);
        t_cost = toc;
        
        stats = eval_equalizer_pam4(ye_use, idxTx, xsym, xm, NumPreamble_TDE, M);
        
        fprintf('  Algo: %s | BER: %.2e | Time: %.2fs\n', algo_id, stats.BER, t_cost);
    end
end

%% ---------------- local functions (Copied from roll_7) ----------------
function [ye_use, idxTx, best_delay, best_offset] = run_equalizer(algo_id, xRx, xTx, xsym, NumPreamble_TDE, M, params)
    clear ye ye_valid valid_idx net

    best_delay = NaN;
    best_offset = NaN;

    switch upper(algo_id)
        case 'FFE'
            [~, ye] = FFE_2pscenter(xRx, xTx, NumPreamble_TDE, params.N1, params.Lambda);
            is_nn = false;
        case 'FNN'
            [ye_valid, ~, valid_idx, best_delay, best_offset] = FNN_FS2pscenter( ...
                xRx, xTx, NumPreamble_TDE, params.FNN_InputLength, params.FNN_HiddenSize, ...
                params.FNN_LR, params.FNN_Epochs, params.FNN_DelayCandidates, params.FNN_OffsetCandidates);
            is_nn = true;
        otherwise
            error('Unknown algorithm: %s', algo_id);
    end

    if is_nn
        idxTx = valid_idx(:);
        if exist('ye_valid', 'var') && ~isempty(ye_valid)
            ye_use = ye_valid(:);
        else
            ye_use = ye(idxTx);
        end
    else
        [off, d0] = align_offset_delay_by_ser(ye, xsym, NumPreamble_TDE, M, -60:60);
        if length(ye) > 1.5 * length(xsym)
            ye_use = ye(off:2:end);
        else
            ye_use = ye(:);
        end
        idxTx = (1:length(ye_use)).' + d0;
    end
end
