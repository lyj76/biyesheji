%% Low-Rank Volterra Test Script V3 (Aggressive)
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
BER_Results = zeros(length(file_list), 2); 

for n1 = 1:length(file_list)
    disp(['Processing File: ', file_list{n1}]);
    load(file_list{n1}, 'ReData');
    ReData = -ReData;

    % Sync
    [TE, ~] = TEFEMMM2(ReData, 1024, 0.3);
    ysync = ReData(1024 + 20 + TE : end);
    ysync = ysync(1 : length(x_shape));
    yt_filter = ysync(1 + N/2 : length(ysync) - N/2);
    xTx = xs;
    xRx = yt_filter;
    
    NumPreamble = 30000; 
    
    %% Config 1: Rank-4, Tap-15, Weighted Loss, No Feedback
    disp('--- Running Config 1: Aggressive (No Feedback) ---');
    Cfg1.Rank = 4;
    Cfg1.Tap = 15;
    Cfg1.LR = [1e-3, 1e-4];
    Cfg1.Epochs = 30;
    Cfg1.Feedback = false;
    Cfg1.WeightedLoss = true;
    
    [ye_1, ~, idx_1] = LowRank_Volterra_Implementation_v3(xRx, xTx, xsym, NumPreamble, Cfg1);
    stats1 = eval_equalizer_pam4(ye_1, idx_1, xsym, xm, NumPreamble, M);
    BER_Results(n1, 1) = stats1.BER;
    disp(['   [Config-1] BER: ', num2str(stats1.BER)]);
    
    %% Config 2: Rank-4, Tap-15, Weighted Loss, WITH Feedback
    disp('--- Running Config 2: Aggressive (WITH Feedback) ---');
    Cfg2 = Cfg1;
    Cfg2.Feedback = true;
    
    [ye_2, ~, idx_2] = LowRank_Volterra_Implementation_v3(xRx, xTx, xsym, NumPreamble, Cfg2);
    stats2 = eval_equalizer_pam4(ye_2, idx_2, xsym, xm, NumPreamble, M);
    BER_Results(n1, 2) = stats2.BER;
    disp(['   [Config-2] BER: ', num2str(stats2.BER)]);
end

disp('=== Final Summary (Aggressive v3) ===');
disp('File | NoFeedback | Feedback');
for i = 1:length(file_list)
    fprintf('%s | %.2e | %.2e\n', file_list{i}, BER_Results(i, 1), BER_Results(i, 2));
end
