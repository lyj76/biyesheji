%% Quick Validation: Linear vs RNNs (High Power Test)
clear;
close all;

%% add paths
addpath('fns');
addpath(fullfile('fns','fns2'));

%% parameters
NumSymbols = 2^18; 
M = 4;
NumPreamble_TDE = 10000;

s = RandStream.create('mt19937ar', 'seed', 529558);
prevStream = RandStream.setGlobalStream(s);

%% PAM4 modulate
[xsym, xm] = PAMSource(M, NumSymbols);
xs = xm;

%% pulse shaping
rolloff = 0.1;
Osamp_factor = 2;
N = 128;
sqrt_ht = rcosdesign(rolloff, N/Osamp_factor, Osamp_factor, 'normal');
sqrt_ht = sqrt_ht ./ max(sqrt_ht);
x_upsamp = upsample(xs, Osamp_factor);
x_shape = conv(sqrt_ht, x_upsamp);
x_shape = x_shape ./ sqrt(mean(abs(x_shape).^2));

%% File Loop
file_list = {'rop3dBm_1.mat', 'rop5dBm_1.mat'};

for f_idx = 1:length(file_list)
    file_name = file_list{f_idx};
    disp(['================================================']);
    disp(['Processing File: ', file_name]);
    
    if ~exist(file_name, 'file'), error('File not found'); end
    load(file_name, 'ReData');
    ReData = -ReData;

    %% Sync
    th = 0.3;
    [TE, FE] = TEFEMMM2(ReData, 1024, th);
    ysync = ReData(1024 + 20 + TE : end);
    ysync = ysync(1 : length(x_shape));
    yt_filter = ysync(1 + N/2 : length(ysync) - N/2);

    xTx = xs;
    xRx = yt_filter;

    %% --- 0. Linear FFE+DFE (Baseline) ---
    disp('Running LE_FFE_DFE (Baseline)...');
    tic;
    params.N1 = 111; params.D1 = 25; params.Lambda = 0.9999; params.scale = M/2;
    [~, ~, ye_lin] = LE_FFE2ps_centerDFE_new(xRx, xTx, NumPreamble_TDE, params.N1, params.D1, params.Lambda, M, params.scale);
    t_lin = toc;

    [off, d0] = align_offset_delay_by_ser(ye_lin, xsym, NumPreamble_TDE, M, -60:60);
    if length(ye_lin) > 1.5 * length(xsym)
        ye_lin_use = ye_lin(off:2:end);
    else
        ye_lin_use = ye_lin(:);
    end
    idx_lin = (1:length(ye_lin_use)).' + d0;

    stats_lin = eval_equalizer_pam4(ye_lin_use, idx_lin, xsym, xm, NumPreamble_TDE, M);
    
    %% --- 0.5. FNN (No Feedback) ---
    disp('Running FNN (No Feedback)...');
    tic;
    [ye_fnn, ~, idx_fnn, d_fnn, p_fnn] = FNN_FS2pscenter(xRx, xTx, NumPreamble_TDE, 101, 64, 0.001, 30, -30:30, [0 1]);
    t_fnn = toc;
    stats_fnn = eval_equalizer_pam4(ye_fnn(idx_fnn), idx_fnn, xsym, xm, NumPreamble_TDE, M);

    %% --- 1. Powerful RNN (Original Version) ---
    disp('Running RNN (Original Powerful Version)...');
    tic;
    
    % Force Delay/Offset based on old successful logs
    if contains(file_name, 'rop3dBm')
        force_delay = 8; force_offset = 1;
    elseif contains(file_name, 'rop5dBm')
        force_delay = 10; force_offset = 1;
    else
        force_delay = []; force_offset = [];
    end
    
    % InputLength=101, Hidden=64, k=2, Epochs=50, Force Delay
    [ye_rnn, ~, idx_rnn, d_rnn, off_rnn] = RNN_Implementation(xRx, xTx, NumPreamble_TDE, 101, 64, 0.001, 50, 2, -30:30, [1 2], [], [], force_delay, force_offset);
    t_rnn = toc;
    stats_rnn = eval_equalizer_pam4(ye_rnn(idx_rnn), idx_rnn, xsym, xm, NumPreamble_TDE, M);

    disp('------------------------------------------------');
    disp(['Summary for ', file_name, ':']);
    disp(['Linear FFE+DFE: ', num2str(stats_lin.BER)]);
    disp(['FNN (No FB):    ', num2str(stats_fnn.BER)]);
    disp(['RNN (Powerful): ', num2str(stats_rnn.BER)]);
end
disp('================================================');