%% RNN Advanced Full Range Test (-1dBm to 5dBm)
clear;
close all;

%% add paths
addpath('fns');
addpath(fullfile('fns','fns2'));

%% parameters
Ft = 200e9;
Osamp_factor = 2;
NumSymbols = 2^18; % 262144 symbols
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

%% Data Files (-1dBm to 5dBm)
% 注意文件名顺序，确保覆盖全范围
file_list = { ...
    'rop-1dBm_1.mat', ...
    'rop0dBm_1.mat', ...
    'rop1dBm_1.mat', ...
    'rop2dBm_1.mat', ...
    'rop3dBm_1.mat', ...
    'rop5dBm_1.mat' ...
};

%% Advanced RNN Parameters
InputLength = 101; % N1
HiddenSize = 32;   
LearningRate = 0.001;
MaxEpochs = 30;    
k_fb = 25;         % Feedback Taps (D1)
NumPreamble_TDE = 10000; % Training Set Size

disp('==========================================================');
disp('   Advanced Residual RNN (WD-AR-RNN) - Full Roll Test');
disp('==========================================================');

BER_Log = zeros(length(file_list), 1);

for i = 1:length(file_list)
    file_name = file_list{i};
    disp(['Processing: ', file_name]);
    
    if ~exist(file_name, 'file')
        warning(['File not found: ', file_name]);
        BER_Log(i) = NaN;
        continue;
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
    
    xTx = xs;         % Length: 262144
    xRx = yt_filter;  % Length: ~524288 (2sps)

    %% Run Advanced RNN
    tic;
    [ye_adv, ~, idx_adv] = RNN_Advanced_Implementation(xRx, xTx, NumPreamble_TDE, ...
        InputLength, HiddenSize, LearningRate, MaxEpochs, k_fb);
    time_cost = toc;
    
    %% Evaluate BER
    % 关键点：这里 eval_equalizer_pam4 会自动忽略前 NumPreamble_TDE 个符号
    % 我们必须确保 idx_adv 和 ye_adv 是对齐的
    
    % ye_adv 是全长输出 (N_total)，非有效位置补零
    % idx_adv 是有效位置索引
    
    % 取出有效部分进行评估
    ye_valid = ye_adv(idx_adv);
    
    stats = eval_equalizer_pam4(ye_valid, idx_adv, xsym, xm, NumPreamble_TDE, M);
    
    BER_Log(i) = stats.BER;
    
    disp(['  -> Time: ', num2str(time_cost, '%.2f'),'s']);
    disp(['  -> BER:  ', num2str(stats.BER)]);
    disp('----------------------------------------------------------');
end

%% Final Summary
disp('Summary Table:');
for i = 1:length(file_list)
    fprintf('%15s : BER = %.5e\n', file_list{i}, BER_Log(i));
end
