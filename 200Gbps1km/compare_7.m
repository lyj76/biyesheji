%% Compare 7 equalizers on PAM4 1km dataset
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
h = fdesign.pulseshaping(Osamp_factor, 'Raised Cosine', 'N,Beta', N, rolloff);
Hd = design(h);
sqrt_ht = Hd.Numerator;
sqrt_ht = sqrt_ht ./ max(sqrt_ht);

x_upsamp = upsample(xs, Osamp_factor);
x_shape = conv(sqrt_ht, x_upsamp);
x_shape = x_shape ./ sqrt(mean(abs(x_shape).^2));

%% data list
file_list = {'rop3dBm_1.mat', 'rop5dBm_1.mat'};

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
SNRall = zeros(length(file_list), length(algo_list));
FNN_BER_train = NaN(length(file_list), 1);
FNN_BER_test = NaN(length(file_list), 1);
FNN_SER_train = NaN(length(file_list), 1);
FNN_SER_test = NaN(length(file_list), 1);
RNN_BER_train = NaN(length(file_list), 1);
RNN_BER_test = NaN(length(file_list), 1);
RNN_SER_train = NaN(length(file_list), 1);
RNN_SER_test = NaN(length(file_list), 1);
FNN_best_delay = NaN(length(file_list), 1);
FNN_best_offset = NaN(length(file_list), 1);
RNN_best_delay = NaN(length(file_list), 1);
RNN_best_offset = NaN(length(file_list), 1);

for n1 = 1:length(file_list)
    disp(['Processing File Index: ', num2str(n1)]);
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

        [ye_use, idxTx, best_delay, best_offset] = run_equalizer(algo_id, xRx, xTx, xsym, NumPreamble_TDE, M, params);

        stats = eval_equalizer_pam4(ye_use, idxTx, xsym, xm, NumPreamble_TDE, M);
        BERall(n1, a) = stats.BER;
        SNRall(n1, a) = stats.SNRdB;

        disp(['File ', num2str(n1), ', Algo ', algo_id, ', BER = ', num2str(stats.BER)]);
        disp(['File ', num2str(n1), ', Algo ', algo_id, ', SNR (dB) = ', num2str(stats.SNRdB)]);

        if strcmpi(algo_id, 'FNN')
            tt = eval_train_test_pam4(ye_use, idxTx, xsym, xm, NumPreamble_TDE, M);
            FNN_BER_train(n1) = tt.BER_train;
            FNN_BER_test(n1) = tt.BER_test;
            FNN_SER_train(n1) = tt.SER_train;
            FNN_SER_test(n1) = tt.SER_test;
            FNN_best_delay(n1) = best_delay;
            FNN_best_offset(n1) = best_offset;
            disp(['File ', num2str(n1), ', FNN train BER = ', num2str(tt.BER_train), ', test BER = ', num2str(tt.BER_test)]);
            disp(['File ', num2str(n1), ', FNN train SER = ', num2str(tt.SER_train), ', test SER = ', num2str(tt.SER_test)]);
            disp(['File ', num2str(n1), ', FNN best_offset = ', num2str(best_offset), ', best_delay = ', num2str(best_delay)]);
            dd = eval_nn_diagnostics(ye_use, idxTx, xsym, xm, NumPreamble_TDE, M);
            disp(['File ', num2str(n1), ', FNN a_train = ', num2str(dd.a_train), ', b_train = ', num2str(dd.b_train), ...
                ', a_test = ', num2str(dd.a_test), ', b_test = ', num2str(dd.b_test)]);
            disp(['File ', num2str(n1), ', FNN mean/std train = ', num2str(dd.mean_train), '/', num2str(dd.std_train), ...
                ', test = ', num2str(dd.mean_test), '/', num2str(dd.std_test)]);
            disp(['File ', num2str(n1), ', FNN corr train = ', num2str(dd.corr_train), ', test = ', num2str(dd.corr_test)]);
            disp(['File ', num2str(n1), ', FNN BER test using train thr = ', num2str(dd.ber_test_train_thr), ...
                ', using test thr = ', num2str(dd.ber_test_test_thr)]);
        elseif strcmpi(algo_id, 'RNN')
            tt = eval_train_test_pam4(ye_use, idxTx, xsym, xm, NumPreamble_TDE, M);
            RNN_BER_train(n1) = tt.BER_train;
            RNN_BER_test(n1) = tt.BER_test;
            RNN_SER_train(n1) = tt.SER_train;
            RNN_SER_test(n1) = tt.SER_test;
            RNN_best_delay(n1) = best_delay;
            RNN_best_offset(n1) = best_offset;
            disp(['File ', num2str(n1), ', RNN train BER = ', num2str(tt.BER_train), ', test BER = ', num2str(tt.BER_test)]);
            disp(['File ', num2str(n1), ', RNN train SER = ', num2str(tt.SER_train), ', test SER = ', num2str(tt.SER_test)]);
            disp(['File ', num2str(n1), ', RNN best_offset = ', num2str(best_offset), ', best_delay = ', num2str(best_delay)]);
            dd = eval_nn_diagnostics(ye_use, idxTx, xsym, xm, NumPreamble_TDE, M);
            disp(['File ', num2str(n1), ', RNN a_train = ', num2str(dd.a_train), ', b_train = ', num2str(dd.b_train), ...
                ', a_test = ', num2str(dd.a_test), ', b_test = ', num2str(dd.b_test)]);
            disp(['File ', num2str(n1), ', RNN mean/std train = ', num2str(dd.mean_train), '/', num2str(dd.std_train), ...
                ', test = ', num2str(dd.mean_test), '/', num2str(dd.std_test)]);
            disp(['File ', num2str(n1), ', RNN corr train = ', num2str(dd.corr_train), ', test = ', num2str(dd.corr_test)]);
            disp(['File ', num2str(n1), ', RNN BER test using train thr = ', num2str(dd.ber_test_train_thr), ...
                ', using test thr = ', num2str(dd.ber_test_test_thr)]);
        end
    end
end

%% summary
avg_BER = mean(BERall, 1, 'omitnan');
avg_SNR = mean(SNRall, 1, 'omitnan');

T = table(algo_list(:), avg_BER(:), avg_SNR(:), 'VariableNames', {'Algo', 'BER', 'SNRdB'});
disp('Average performance over files:');
disp(T);

disp('FNN/RNN train-test BER per file:');
T_nn = table(file_list(:), ...
    FNN_BER_train, FNN_BER_test, FNN_SER_train, FNN_SER_test, ...
    RNN_BER_train, RNN_BER_test, RNN_SER_train, RNN_SER_test, ...
    FNN_best_offset, FNN_best_delay, RNN_best_offset, RNN_best_delay, ...
    'VariableNames', {'File', ...
    'FNN_BER_train', 'FNN_BER_test', 'FNN_SER_train', 'FNN_SER_test', ...
    'RNN_BER_train', 'RNN_BER_test', 'RNN_SER_train', 'RNN_SER_test', ...
    'FNN_best_offset', 'FNN_best_delay', 'RNN_best_offset', 'RNN_best_delay'});
disp(T_nn);

%% plots for paper
figure('Name', 'Average BER (log scale)');
semilogy(1:length(algo_list), avg_BER, 'o-', 'LineWidth', 1.5);
grid on;
set(gca, 'XTick', 1:length(algo_list), 'XTickLabel', algo_list);
xlabel('Algorithm');
ylabel('BER (log scale)');
title('Average BER over files');

figure('Name', 'Average SNR');
bar(avg_SNR);
grid on;
set(gca, 'XTick', 1:length(algo_list), 'XTickLabel', algo_list);
xlabel('Algorithm');
ylabel('SNR (dB)');
title('Average SNR over files');

figure('Name', 'Per-file BER (log scale)');
ber_plot = BERall.'; % [7 x 2] 矩阵

% 1. 获取 bar 对象的句柄 b
b = bar(ber_plot, 'grouped');
grid on;

% 2. 关键修复：设置对数坐标和柱状图的基准底线
set(gca, 'YScale', 'log');
% 强制让柱子从 1e-6 开始往上长，而不是从 0 或 1 开始
% 这样 1e-4 的柱子就会显示为从 1e-6 到 1e-4 的高度
for i = 1:length(b)
    b(i).BaseValue = 1e-6; 
end

% 3. 调整坐标轴范围，确保显示美观
ylim([1e-6 1]); 

set(gca, 'XTick', 1:length(algo_list), 'XTickLabel', algo_list);
xlabel('Algorithm');
ylabel('BER (log scale)');
legend(file_list, 'Location', 'northeast'); % 改一下图例位置防止遮挡
title('BER per file');

figure('Name', 'Per-file SNR');
bar(SNRall.');
grid on;
set(gca, 'XTick', 1:length(algo_list), 'XTickLabel', algo_list);
xlabel('Algorithm');
ylabel('SNR (dB)');
legend(file_list, 'Location', 'best');
title('SNR per file');

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
            [ye_valid, net, valid_idx, best_delay, best_offset] = FNN_Implementation( ...
                xRx, xTx, NumPreamble_TDE, params.FNN_InputLength, params.FNN_HiddenSize, ...
                params.FNN_LR, params.FNN_Epochs, params.FNN_DelayCandidates, params.FNN_OffsetCandidates);
            is_nn = true;
        case 'RNN'
            [ye, net, valid_idx, best_delay, best_offset] = RNN_Implementation(xRx, xTx, NumPreamble_TDE, params.RNN_InputLength, params.RNN_HiddenSize, params.RNN_LR, params.RNN_Epochs, params.RNN_k, params.RNN_DelayCandidates, params.RNN_OffsetCandidates);
            is_nn = true;
        otherwise
            error('Unknown algo_id: %s', algo_id);
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

function stats = eval_train_test_pam4(ye, idxTx, xsym, xm, NumPreamble_TDE, M)
    idxTx = idxTx(:);
    ye = ye(:);

    m = idxTx >= 1 & idxTx <= length(xsym) & isfinite(ye);
    idxTx = idxTx(m);
    ye = ye(m);

    idx_train = idxTx(idxTx <= NumPreamble_TDE);
    idx_test  = idxTx(idxTx >  NumPreamble_TDE);

    if isempty(idx_train) || isempty(idx_test)
        stats.BER_train = NaN;
        stats.BER_test = NaN;
        stats.SER_train = NaN;
        stats.SER_test = NaN;
        return;
    end

    y_train = ye(idxTx <= NumPreamble_TDE);
    y_test  = ye(idxTx >  NumPreamble_TDE);

    xm_train = xm(idx_train);
    xm_test = xm(idx_test);

    A = [double(y_train), ones(length(y_train),1)];
    p = A \ double(xm_train);
    a = p(1);
    b = p(2);

    xhat_train = a * double(y_train) + b;
    xhat_test = a * double(y_test) + b;

    ysym_train = pamdemod(xhat_train, M, 0, 'gray');
    ysym_test = pamdemod(xhat_test, M, 0, 'gray');
    xsym_train = xsym(idx_train);
    xsym_test = xsym(idx_test);

    [~, ber_tr] = biterr(ysym_train, xsym_train, log2(M));
    [~, ber_te] = biterr(ysym_test, xsym_test, log2(M));
    [~, ser_tr] = symerr(ysym_train, xsym_train);
    [~, ser_te] = symerr(ysym_test, xsym_test);

    stats.BER_train = ber_tr;
    stats.BER_test = ber_te;
    stats.SER_train = ser_tr;
    stats.SER_test = ser_te;
end

function dd = eval_nn_diagnostics(ye, idxTx, xsym, xm, NumPreamble_TDE, M)
    idxTx = idxTx(:);
    ye = ye(:);

    m = idxTx >= 1 & idxTx <= length(xsym) & isfinite(ye);
    idxTx = idxTx(m);
    ye = ye(m);

    idx_train = idxTx(idxTx <= NumPreamble_TDE);
    idx_test  = idxTx(idxTx >  NumPreamble_TDE);

    if isempty(idx_train) || isempty(idx_test)
        dd.a_train = NaN; dd.b_train = NaN;
        dd.a_test = NaN; dd.b_test = NaN;
        dd.mean_train = NaN; dd.std_train = NaN;
        dd.mean_test = NaN; dd.std_test = NaN;
        dd.corr_train = NaN; dd.corr_test = NaN;
        dd.ber_test_train_thr = NaN; dd.ber_test_test_thr = NaN;
        return;
    end

    y_train = ye(idxTx <= NumPreamble_TDE);
    y_test  = ye(idxTx >  NumPreamble_TDE);

    xm_train = xm(idx_train);
    xm_test = xm(idx_test);

    A_tr = [double(y_train), ones(length(y_train),1)];
    p_tr = A_tr \ double(xm_train);
    a_tr = p_tr(1);
    b_tr = p_tr(2);

    A_te = [double(y_test), ones(length(y_test),1)];
    p_te = A_te \ double(xm_test);
    a_te = p_te(1);
    b_te = p_te(2);

    xhat_train = a_tr * double(y_train) + b_tr;
    xhat_test_tr = a_tr * double(y_test) + b_tr;
    xhat_test_te = a_te * double(y_test) + b_te;

    [levels_tr, thr_tr] = find_levels_thr(xhat_train, M);
    [levels_te, thr_te] = find_levels_thr(xhat_test_te, M);

    ysym_test_tr = pamdemod(slice_levels(xhat_test_tr, levels_tr, thr_tr), M, 0, 'gray');
    ysym_test_te = pamdemod(slice_levels(xhat_test_te, levels_te, thr_te), M, 0, 'gray');
    xsym_test = xsym(idx_test);

    [~, ber_tr_thr] = biterr(ysym_test_tr, xsym_test, log2(M));
    [~, ber_te_thr] = biterr(ysym_test_te, xsym_test, log2(M));

    dd.a_train = a_tr;
    dd.b_train = b_tr;
    dd.a_test = a_te;
    dd.b_test = b_te;
    dd.mean_train = mean(y_train);
    dd.std_train = std(y_train);
    dd.mean_test = mean(y_test);
    dd.std_test = std(y_test);
    dd.corr_train = corr_simple(y_train, xm_train);
    dd.corr_test = corr_simple(y_test, xm_test);
    dd.ber_test_train_thr = ber_tr_thr;
    dd.ber_test_test_thr = ber_te_thr;
end

function [levels, thr] = find_levels_thr(x, M)
    x = double(x(:));
    [~, C] = kmeans(x, M, 'Replicates', 3);
    levels = sort(C(:)).';
    thr = (levels(1:M-1) + levels(2:M)) / 2;
end

function yq = slice_levels(y, levels, thr)
    y = double(y);
    yq = zeros(size(y));
    yq(y < thr(1)) = levels(1);
    for k = 2:length(thr)
        yq(y >= thr(k-1) & y < thr(k)) = levels(k);
    end
    yq(y >= thr(end)) = levels(end);
end

function c = corr_simple(a, b)
    a = double(a(:));
    b = double(b(:));
    m = isfinite(a) & isfinite(b);
    a = a(m); b = b(m);
    if numel(a) < 2
        c = NaN;
        return;
    end
    a = a - mean(a);
    b = b - mean(b);
    c = (a' * b) / sqrt((a' * a) * (b' * b) + eps);
end
