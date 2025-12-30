%% Roll-off style comparison: BER vs noise (dB) for 7 equalizers
clear;
close all;

%% add paths
addpath('fns');
addpath(fullfile('fns','fns2'));

%% parameters
Ft = 200e9;
Osamp_factor = 2;
NumSymbols = 2^18;
NumPreamble = 0;
NumSym_total = NumSymbols + NumPreamble;
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

%% data list (different dB)
file_list = { ...
    'rop-1dBm_1.mat', ...
    'rop0dBm_1.mat', ...
    'rop1dBm_1.mat', ...
    'rop2dBm_1.mat', ...
    'rop3dBm_1.mat', ...
    'rop5dBm_1.mat' ...
};

%% parse dB from filenames
noise_dB = zeros(size(file_list));
for i = 1:numel(file_list)
    tok = regexp(file_list{i}, 'rop(-?\d+)dBm', 'tokens', 'once');
    if isempty(tok)
        error('Cannot parse dB from file name: %s', file_list{i});
    end
    noise_dB(i) = str2double(tok{1});
end
[noise_dB, sort_idx] = sort(noise_dB);
file_list = file_list(sort_idx);

%% equalizer parameters
params.N1 = 111;
params.N2 = 21;
params.WL = 1;
params.D1 = 25;
params.D2 = 3;
params.WD = 1;
params.K_Lin = 18;
params.K_Vol = 90;
params.Lambda = 0.9999;
params.scale = M/2;

%% NN parameters
params.FNN_InputLength = 91;
params.FNN_HiddenSize = 64;
params.FNN_LR = 0.001;
params.FNN_Epochs = 50;
params.FNN_DelayCandidates = -30:30;
params.FNN_OffsetCandidates = [1 2];

params.RNN_InputLength = 61;
params.RNN_HiddenSize = 64;
params.RNN_LR = 0.001;
params.RNN_Epochs = 15;
params.RNN_k = 25;
params.RNN_DelayCandidates = [8 10];
params.RNN_OffsetCandidates = [1 2];

%% algo list
algo_list = { ...
    'FFE', ...
    'VNLE', ...
    'LE_FFE_DFE', ...
    'DP_VFFE_VDFE', ...
    'CLUT_VDFE', ...
    'FNN', ...
    'RNN' ...
};

BERall = zeros(length(file_list), length(algo_list));

for n1 = 1:length(file_list)
    disp(['Processing File: ', file_list{n1}]);
    load(file_list{n1}, 'ReData')

    ReData = -ReData;

    %% synchronization
    th = 0.3;
    [TE, FE] = TEFEMMM2(ReData, 1024, th);
    abc = TE + 0;
    TE = abc;

    ysync = ReData(1024 + 20 + TE : end);
    ysync = ysync(1 : length(x_shape));

    %% match filtering
    yt_filter = ysync(1 + N/2 : length(ysync) - N/2);

    %% equalizer inputs
    xTx = xs;
    xRx = yt_filter;
    NumPreamble_TDE = 10000;

    for a = 1:length(algo_list)
        algo_id = algo_list{a};

        [ye_use, idxTx] = run_equalizer(algo_id, xRx, xTx, xsym, NumPreamble_TDE, M, params);

        stats = eval_equalizer_pam4(ye_use, idxTx, xsym, xm, NumPreamble_TDE, M);
        BERall(n1, a) = stats.BER;

        disp(['File ', file_list{n1}, ', Algo ', algo_id, ', BER = ', num2str(stats.BER)]);
    end
end

%% plot: BER vs noise dB
figure('Name', 'BER vs Noise (dB)');
hold on;
for a = 1:length(algo_list)
    semilogy(noise_dB, BERall(:, a), 'o-', 'LineWidth', 1.5);
end
grid on;
xlabel('Noise (dB)');
ylabel('BER (log scale)');
title('BER vs Noise (dB) for 7 Equalizers');
legend(algo_list, 'Location', 'best');

%% ---------------- local functions ----------------
function [ye_use, idxTx, best_delay, best_offset] = run_equalizer(algo_id, xRx, xTx, xsym, NumPreamble_TDE, M, params)
    clear ye ye_valid valid_idx net

    best_delay = NaN;
    best_offset = NaN;

    switch upper(algo_id)
        case 'FFE'
            [~, ye] = FFE_2pscenter(xRx, xTx, NumPreamble_TDE, params.N1, params.Lambda);
            is_nn = false;
        case 'VNLE'
            [~, ye] = VNLE2_2pscenter(xRx, xTx, NumPreamble_TDE, params.N1, params.N2, params.Lambda, params.WL);
            is_nn = false;
        case 'LE_FFE_DFE'
            [~, ~, ye] = LE_FFE2ps_centerDFE_new(xRx, xTx, NumPreamble_TDE, params.N1, params.D1, params.Lambda, M, params.scale);
            is_nn = false;
        case 'DP_VFFE_VDFE'
            [~, ~, ye] = DP_VFFE2pscenter_VDFE(xRx, xTx, NumPreamble_TDE, params.N1, params.N2, params.D1, params.D2, params.Lambda, params.WL, params.WD, M, params.scale);
            is_nn = false;
        case 'CLUT_VDFE'
            [~, ye] = CLUT_VDFE_Implementation(xRx, xTx, NumPreamble_TDE, params.N1, params.N2, params.D1, params.D2, params.WL, params.WD, M, params.K_Lin, params.K_Vol, params.Lambda);
            is_nn = false;
        case 'FNN'
            [ye_valid, ~, valid_idx, best_delay, best_offset] = FNN_Implementation( ...
                xRx, xTx, NumPreamble_TDE, params.FNN_InputLength, params.FNN_HiddenSize, ...
                params.FNN_LR, params.FNN_Epochs, params.FNN_DelayCandidates, params.FNN_OffsetCandidates);
            is_nn = true;
        case 'RNN'
            [ye, ~, valid_idx, best_delay, best_offset] = RNN_Implementation( ...
                xRx, xTx, NumPreamble_TDE, params.RNN_InputLength, params.RNN_HiddenSize, ...
                params.RNN_LR, params.RNN_Epochs, params.RNN_k, params.RNN_DelayCandidates, params.RNN_OffsetCandidates);
            is_nn = true;
        otherwise
            error('Unknown algorithm: %s', algo_id);
    end

    if is_nn
        if ~exist('valid_idx', 'var') || isempty(valid_idx)
            error('valid_idx missing for NN algorithm: %s', algo_id);
        end
        idxTx = valid_idx(:);
        if exist('ye_valid', 'var') && ~isempty(ye_valid)
            ye_use = ye_valid(:);
        else
            ye_use = ye(idxTx);
        end
    else
        if ~exist('ye', 'var') || isempty(ye)
            error('ye missing for classical algorithm: %s', algo_id);
        end
        [off, d0] = align_offset_delay_by_ser(ye, xsym, NumPreamble_TDE, M, -60:60);
        if length(ye) > 1.5 * length(xsym)
            ye_use = ye(off:2:end);
        else
            ye_use = ye(:);
        end
        idxTx = (1:length(ye_use)).' + d0;
    end
end
