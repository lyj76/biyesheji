%% PAM4 main - FFE vs FNN Debug
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

%% Test File: rop3dBm_1.mat
file_name = 'rop3dBm_1.mat';
if ~exist(file_name, 'file')
    error(['File not found: ', file_name]);
end
load(file_name, 'ReData');
ReData = -ReData;

%% Synchronization
th = 0.3;
[TE, FE] = TEFEMMM2(ReData, 1024, th);
ysync = ReData(1024 + 20 + TE : end);
ysync = ysync(1 : length(x_shape));

%% Match Filtering
yt_filter = ysync(1 + N/2 : length(ysync) - N/2);
xTx = xs;
xRx = yt_filter;

NumPreamble_TDE = 10000; % Training samples

%% --- 1. Run FFE (Baseline) ---
disp('------------------------------------------------');
disp('Running FFE (Baseline)...');
N1 = 111;
Lambda = 0.9999;
tic;
[hffe, ye_ffe] = FFE_2pscenter(xRx, xTx, NumPreamble_TDE, N1, Lambda);
time_ffe = toc;
disp(['FFE Time: ', num2str(time_ffe), 's']);

% FFE Alignment
[off_ffe, d0_ffe] = align_offset_delay_by_ser(ye_ffe, xsym, NumPreamble_TDE, M, -60:60);
if length(ye_ffe) > 1.5 * length(xsym)
    ye_use_ffe = ye_ffe(off_ffe:2:end);
else
    ye_use_ffe = ye_ffe(:);
end
idxTx_ffe = (1:length(ye_use_ffe)).' + d0_ffe;

stats_ffe = eval_equalizer_pam4(ye_use_ffe, idxTx_ffe, xsym, xm, NumPreamble_TDE, M);
disp(['FFE BER: ', num2str(stats_ffe.BER)]);
disp(['FFE Optimal Delay (d0): ', num2str(d0_ffe)]);

%% --- 2. Run FNN (Standard) ---
disp('------------------------------------------------');
disp('Running FNN (Standard)...');
InputLength = 111; 
HiddenSize = 16;
LearningRate = 0.001;
MaxEpochs = 50;

tic;
[ye_fnn, ~, valid_idx_fnn, best_delay_fnn, best_offset_fnn] = FNN_Implementation(xRx, xTx, NumPreamble_TDE, ...
    InputLength, HiddenSize, LearningRate, MaxEpochs);
time_fnn = toc;
disp(['FNN Time: ', num2str(time_fnn), 's']);
disp(['FNN Found Delay: ', num2str(best_delay_fnn), ', Offset: ', num2str(best_offset_fnn)]);

idxTx_fnn = valid_idx_fnn(:);
ye_use_fnn = ye_fnn(:);

stats_fnn = eval_equalizer_pam4(ye_use_fnn, idxTx_fnn, xsym, xm, NumPreamble_TDE, M);
disp(['FNN (Standard) BER: ', num2str(stats_fnn.BER)]);

disp('------------------------------------------------');
