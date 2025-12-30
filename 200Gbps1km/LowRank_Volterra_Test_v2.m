%% Low-Rank Volterra Test Script V2 (Robust)
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
h = fdesign.pulseshaping(Osamp_factor, 'Raised Cosine', 'N,Beta', N, rolloff);
Hd = design(h);
sqrt_ht = Hd.Numerator;
sqrt_ht = sqrt_ht ./ max(sqrt_ht);

x_upsamp = upsample(xs, Osamp_factor);
x_shape = conv(sqrt_ht, x_upsamp);
x_shape = x_shape ./ sqrt(mean(abs(x_shape).^2));

%% File List
file_list = {'rop3dBm_1.mat', 'rop5dBm_1.mat'}; 
BER_Results = zeros(length(file_list), 2);

for n1 = 1:length(file_list)
    disp(['Processing File: ', file_list{n1}]);
    load(file_list{n1}, 'ReData');
    ReData = -ReData;

    % Synchronization
    th = 0.3;
    [TE, FE] = TEFEMMM2(ReData, 1024, th);
    ysync = ReData(1024 + 20 + TE : end);
    ysync = ysync(1 : length(x_shape));

    % Match Filtering
    yt_filter = ysync(1 + N/2 : length(ysync) - N/2);
    xTx = xs;
    xRx = yt_filter;
    
    NumPreamble_TDE = 30000; 
    
        %% Phase 1: Volterra Residual (No Feedback)
        disp('--- Running Phase 1: Robust Volterra (No Feedback) ---');
        
        % Rank=8, LR=[5e-3, 5e-4], Epochs=50
        [ye_1, ~, idx_1] = LowRank_Volterra_Implementation_v2(xRx, xTx, xsym, NumPreamble_TDE, ...
            8, [5e-3, 5e-4], 50, false);
        
        stats1 = eval_equalizer_pam4(ye_1, idx_1, xsym, xm, NumPreamble_TDE, M);
        BER_Results(n1, 1) = stats1.BER;
        disp(['   [Volterra-NF] BER: ', num2str(stats1.BER), ', SNR: ', num2str(stats1.SNRdB)]);
        
        %% Phase 2: Volterra Residual (WITH Feedback)
        disp('--- Running Phase 2: Robust Volterra (WITH Feedback) ---');
        
        % Rank=8, LR=[5e-3, 5e-4], Epochs=50, Feedback=true
        [ye_2, ~, idx_2] = LowRank_Volterra_Implementation_v2(xRx, xTx, xsym, NumPreamble_TDE, ...
            8, [5e-3, 5e-4], 50, true);        
    stats2 = eval_equalizer_pam4(ye_2, idx_2, xsym, xm, NumPreamble_TDE, M);
    BER_Results(n1, 2) = stats2.BER;
    disp(['   [Volterra-FB] BER: ', num2str(stats2.BER), ', SNR: ', num2str(stats2.SNRdB)]);
end

disp('=== Final Summary ===');
disp('File | NoFeedback(12p) | Feedback(16p)');
for i = 1:length(file_list)
    fprintf('%s | %.2e | %.2e\n', file_list{i}, BER_Results(i, 1), BER_Results(i, 2));
end
