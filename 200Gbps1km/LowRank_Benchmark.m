%% Low-Rank Volterra Benchmark
clear;
close all;

%% add paths
addpath('fns');
addpath(fullfile('fns','fns2'));

%% parameters
M = 4;
NumSymbols = 2^18;
[xsym, xm] = PAMSource(M, NumSymbols);
xsym = xsym(:);
xs = xm;

% Pulse Shaping
rolloff = 0.1; N = 128; Osamp_factor = 2;
h = fdesign.pulseshaping(Osamp_factor, 'Raised Cosine', 'N,Beta', N, rolloff);
Hd = design(h); sqrt_ht = Hd.Numerator; sqrt_ht = sqrt_ht ./ max(sqrt_ht);
x_upsamp = upsample(xs, Osamp_factor);
x_shape = conv(sqrt_ht, x_upsamp); x_shape = x_shape ./ sqrt(mean(abs(x_shape).^2));

%% File List
file_list = {'rop3dBm_1.mat', 'rop5dBm_1.mat'};
Results = table();

for n1 = 1:length(file_list)
    filename = file_list{n1};
    disp(['=== Processing: ', filename, ' ===']);
    load(filename, 'ReData');
    ReData = -ReData;

    % Sync
    [TE, ~] = TEFEMMM2(ReData, 1024, 0.3);
    ysync = ReData(1024 + 20 + TE : end);
    ysync = ysync(1 : length(x_shape));
    yt_filter = ysync(1 + N/2 : length(ysync) - N/2);
    xTx = xs;
    xRx = yt_filter;
    NumPreamble = 30000; 

    %% 1. Standard FFE (Baseline)
    disp('   Running FFE...');
    N1 = 111; Lambda = 0.9999;
    [~, ye_ffe] = FFE_2pscenter(xRx, xTx, NumPreamble, N1, Lambda);
    
    [off, d0] = align_offset_delay_by_ser(ye_ffe, xsym, NumPreamble, M, -60:60);
    if length(ye_ffe) > 1.5 * length(xsym), ye_ffe = ye_ffe(off:2:end); end
    idx_ffe = (1:length(ye_ffe))' + d0;
    
    stats_ffe = eval_equalizer_pam4(ye_ffe, idx_ffe, xsym, xm, NumPreamble, M);
    
    %% 2. VNLE (Strong Baseline)
    disp('   Running VNLE (Volterra)...');
    N2 = 21; WL = 1; % Classic Volterra params
    [~, ye_vnle] = VNLE2_2pscenter(xRx, xTx, NumPreamble, N1, N2, Lambda, WL);
    
    [off, d0] = align_offset_delay_by_ser(ye_vnle, xsym, NumPreamble, M, -60:60);
    if length(ye_vnle) > 1.5 * length(xsym), ye_vnle = ye_vnle(off:2:end); end
    idx_vnle = (1:length(ye_vnle))' + d0;
    
    stats_vnle = eval_equalizer_pam4(ye_vnle, idx_vnle, xsym, xm, NumPreamble, M);

    %% 3. Low-Rank Volterra (Your Algorithm)
    disp('   Running Low-Rank Volterra (Rank 8, Tap 15)...');
    % Rank=8, LR=[5e-3, 5e-4], Epochs=30, No Feedback (since it didn't help)
    % Using Implementation_v2 (the reliable one)
    [ye_lr, params_lr, idx_lr] = LowRank_Volterra_Implementation_v2(xRx, xTx, xsym, NumPreamble, ...
        8, [5e-3, 5e-4], 30, false);
    
    stats_lr = eval_equalizer_pam4(ye_lr, idx_lr, xsym, xm, NumPreamble, M);
    
    %% Record Results
    % Params Estimate:
    % FFE: 111
    % VNLE: 111 + N2 (depends on structure, usually N2*something) ~150-200
    % LowRank: 111 (Backbone) + (15*8 + 8) (Residual) = 239 (Wait, Rank 8 is heavy?)
    % Let's optimize Rank later if needed.
    
    new_row = {filename, ...
        stats_ffe.BER, stats_vnle.BER, stats_lr.BER, ...
        stats_ffe.SNRdB, stats_vnle.SNRdB, stats_lr.SNRdB};
    Results = [Results; new_row];
end

Results.Properties.VariableNames = {'File', 'BER_FFE', 'BER_VNLE', 'BER_LowRank', 'SNR_FFE', 'SNR_VNLE', 'SNR_LowRank'};
disp(Results);

% Calculate Gap
gap_vnle = (Results.BER_LowRank - Results.BER_VNLE) ./ Results.BER_VNLE * 100;
disp('Gap to VNLE (%):');
disp(gap_vnle);
