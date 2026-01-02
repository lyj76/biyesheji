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
    fullfile('data','rop-1dBm_1.mat'), ...
    fullfile('data','rop0dBm_1.mat'), ...
    fullfile('data','rop1dBm_1.mat'), ...
    fullfile('data','rop2dBm_1.mat'), ...
    fullfile('data','rop3dBm_1.mat'), ...
    fullfile('data','rop5dBm_1.mat') ...
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
params.FNN_InputLength = 111;
params.FNN_HiddenSize = 32;
params.FNN_LR = 0.001;
params.FNN_Epochs = 50;
params.FNN_DelayCandidates = -30:30;
params.FNN_OffsetCandidates = [1 2];

params.RNN_InputLength = 101;
params.RNN_HiddenSize = 64;
params.RNN_LR = 0.001;
params.RNN_Epochs = 50;
params.RNN_k = 25; % Increased from 2 to 25 to match DFE length for 200G
params.RNN_DelayCandidates = -30:30;
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

%% plot: BER vs ROP (dBm)
figure('Name', 'BER vs ROP');

% Color Palette (High Contrast)
% 1:FFE(Blue), 2:VNLE(Red), 3:LE_FFE_DFE(Yellow), 4:DP_VFFE(Purple)
% 5:CLUT(Green), 6:FNN(Cyan), 7:RNN(Black)
colors = [
    0 0.4470 0.7410;  % Blue
    0.8500 0.3250 0.0980; % Red
    0.9290 0.6940 0.1250; % Yellow
    0.4940 0.1840 0.5560; % Purple
    0.4660 0.6740 0.1880; % Green
    0.3010 0.7450 0.9330; % Cyan
    0 0 0                 % Black (RNN)
];
markers = {'o-', 's-', 'd-', '^-', 'v-', '>-', 'p-'};

% Handle BER=0 for log plot
BER_plot = BERall;
min_val = 1e-6; % Floor for 0 BER
BER_plot(BER_plot == 0) = min_val; 

% Direct Line Plot (No Spline)
h_plots = gobjects(length(algo_list), 1); % Store handles for legend

hold on;
for a = 1:length(algo_list)
    y_data = BERall(:, a);
    y_data(y_data == 0) = min_val; 
    
    % Plot straight line with markers (Capture handle for legend)
    h_plots(a) = semilogy(noise_dB, y_data, markers{a}, ...
        'Color', colors(a,:), 'LineWidth', 1.5, 'MarkerSize', 7, ...
        'MarkerFaceColor', colors(a,:));
end

% Add baseline threshold line (e.g. HD-FEC limit 3.8e-3)
yline(3.8e-3, '--k', 'HD-FEC (3.8e-3)', 'LabelHorizontalAlignment', 'left');

grid on;
xlabel('Received Optical Power (dBm)');
ylabel('BER (log scale)');
title('BER Performance vs ROP');
legend(h_plots, algo_list, 'Location', 'northeast', 'FontSize', 10);

% Smart Y-limit
max_ber = max(BERall(:));
if max_ber == 0, max_ber = 1e-2; end
ylim([min_val max_ber * 1.1]); % Tight adaptive limit (10% margin)

% Add annotation for 0 BER
text(max(noise_dB)-1, min_val*1.2, 'Floor: BER=0', 'FontSize', 8, 'Color', 'k');

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
            [ye_valid, ~, valid_idx, best_delay, best_offset] = FNN_FS2pscenter( ...
                xRx, xTx, NumPreamble_TDE, params.FNN_InputLength, params.FNN_HiddenSize, ...
                params.FNN_LR, params.FNN_Epochs, params.FNN_DelayCandidates, params.FNN_OffsetCandidates);
            is_nn = true;
        case 'RNN'
            %correct RNN
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
