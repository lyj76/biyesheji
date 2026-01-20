%% Roll-off style comparison: 1 SPS vs 2 SPS (FSE)
% Specifically designed to show Phase Sensitivity of T-spaced Equalizer
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

% Fixed Seed
s = RandStream.create('mt19937ar', 'seed', 529558);
prevStream = RandStream.setGlobalStream(s);

%% PAM4 modulate
[xsym, xm] = PAMSource(M, NumSymbols);
xsym = xsym(:);
xm = xm(:);
xs = xm;

%% pulse shaping (Tx)
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

%% parse dB
noise_dB = zeros(size(file_list));
for i = 1:numel(file_list)
    tok = regexp(file_list{i}, 'rop(-?\d+)dBm', 'tokens', 'once');
    noise_dB(i) = str2double(tok{1});
end
[noise_dB, sort_idx] = sort(noise_dB);
file_list = file_list(sort_idx);

%% equalizer parameters
% 1 SPS FFE length (half of 2 SPS)
params.N1_1sps = 31; 
% 2 SPS FFE length
params.N1_2sps = 61;

params.Lambda = 0.9999;
NumPreamble = 10000;

%% Storage
BER_Phase1 = zeros(length(file_list), 1);
BER_Phase2 = zeros(length(file_list), 1);
BER_FSE    = zeros(length(file_list), 1);

for n1 = 1:length(file_list)
    disp(['Processing File: ', file_list{n1}]);
    load(file_list{n1}, 'ReData')
    ReData = -ReData;

    %% synchronization (Coarse)
    th = 0.3;
    [TE, FE] = TEFEMMM2(ReData, 1024, th);
    ysync = ReData(1024 + 20 + TE : end);
    ysync = ysync(1 : length(x_shape));
    % Match Filtering is usually implicit in FSE, but for T-spaced it's critical.
    % However, to keep fair comparison on "Architecture", we feed same signal.
    % Ideally, T-spaced needs a dedicated analog filter or digital match filter before decimation.
    % Here we use the same filtered signal as FSE.
    yt_filter = ysync(1 + N/2 : length(ysync) - N/2);
    
    xRx_2sps = yt_filter;
    xTx = xs;

    %% --- 1. T-Spaced (Phase 1) ---
    % Naive Decimation: Odd samples
    xRx_p1 = xRx_2sps(1:2:end);
    [ye_p1, idx_p1] = FFE_1sps(xRx_p1, xTx, NumPreamble, params.N1_1sps, params.Lambda);
    stats1 = eval_equalizer_pam4(ye_p1, idx_p1, xsym, xm, NumPreamble, M);
    BER_Phase1(n1) = stats1.BER;
    
    %% --- 2. T-Spaced (Phase 2) ---
    % Naive Decimation: Even samples
    xRx_p2 = xRx_2sps(2:2:end);
    [ye_p2, idx_p2] = FFE_1sps(xRx_p2, xTx, NumPreamble, params.N1_1sps, params.Lambda);
    stats2 = eval_equalizer_pam4(ye_p2, idx_p2, xsym, xm, NumPreamble, M);
    BER_Phase2(n1) = stats2.BER;
    
    %% --- 3. Fractionally Spaced (FSE 2sps) ---
    [~, ye_fse] = FFE_2pscenter(xRx_2sps, xTx, NumPreamble, params.N1_2sps, params.Lambda);
    % Alignment for 2sps output
    [off, d0] = align_offset_delay_by_ser(ye_fse, xsym, NumPreamble, M, -60:60);
    if length(ye_fse) > 1.5 * length(xsym)
        ye_use = ye_fse(off:2:end);
    else
        ye_use = ye_fse(:);
    end
    idxTx_fse = (1:length(ye_use)).' + d0;
    
    stats_fse = eval_equalizer_pam4(ye_use, idxTx_fse, xsym, xm, NumPreamble, M);
    BER_FSE(n1) = stats_fse.BER;

    disp(['  Phase 1 BER: ', num2str(stats1.BER, '%.2e'), ...
          ' | Phase 2 BER: ', num2str(stats2.BER, '%.2e'), ...
          ' | FSE BER: ', num2str(stats_fse.BER, '%.2e')]);
end

%% Plot Comparison
figure('Name', 'T-spaced Phase Sensitivity vs FSE');

% Helper to log10
to_log = @(x) log10(max(x, 1e-7));

plot(noise_dB, to_log(BER_Phase1), 'r^--', 'LineWidth', 1.5, 'DisplayName', 'T-spaced (Phase 1)');
hold on;
plot(noise_dB, to_log(BER_Phase2), 'bs--', 'LineWidth', 1.5, 'DisplayName', 'T-spaced (Phase 2)');
plot(noise_dB, to_log(BER_FSE),    'ko-',  'LineWidth', 2.0, 'DisplayName', 'FSE (2-sps)');

yline(log10(3.8e-3), '--k', 'HD-FEC');
grid on; 
xlabel('Received Optical Power (dBm)');
ylabel('log10(BER)');
title('Impact of Sampling Phase on BER');
legend('Location', 'southwest');
ylim([-5.5 -1]); % 1e-5.5 to 1e-1

