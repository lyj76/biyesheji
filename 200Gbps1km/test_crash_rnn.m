%% Crash Test
clear; close all;
addpath('fns'); addpath(fullfile('fns','fns2'));

NumSym = 20000; M = 4; NumPreamble = 5000;
SNR_Sweep = 14:-2:0; 
rng(123);

tx_sym = randi([0 M-1], NumSym, 1);
tx_amp = pammod(tx_sym, M, 0, 'gray');
tx_amp = tx_amp / std(tx_amp);
tx_upsamp = upsample(tx_amp, 2);

h_isi = [0.05, 0.2, 0.5, 1.0, 0.5, 0.2, 0.05]; 
h_isi = h_isi / norm(h_isi);
rx_lin = conv(tx_upsamp, h_isi, 'same');
rx_nl = rx_lin + 0.1 * (rx_lin.^2) - 0.05 * (rx_lin.^3);

ber_wd = zeros(size(SNR_Sweep));
ber_ar = zeros(size(SNR_Sweep));
ber_ffe = zeros(size(SNR_Sweep));

disp('=== RNN CRASH TEST ===');

for i = 1:length(SNR_Sweep)
    snr = SNR_Sweep(i);
    rx_noisy = awgn(rx_nl, snr, 'measured');
    xRx = rx_noisy; xTx = tx_amp;
    
    % FFE
    [~, ye_ffe] = FFE_2pscenter(xRx, xTx, NumPreamble, 51, 0.999);
    [off, d0] = align_offset_delay_by_ser(ye_ffe, tx_sym, NumPreamble, M, -20:20);
    ye_ffe = ye_ffe(off:2:end); idx_ffe = (1:length(ye_ffe)).' + d0;
    ber_ffe(i) = eval_ber_local(ye_ffe, idx_ffe, tx_sym, NumPreamble, M);

    % RNN_WD
    [ye_wd, ~, idx_wd] = RNN_WD_Implementation(xRx, xTx, NumPreamble, 51, 32, 0.001, 20, 10, -20:20, [1 2]);
    ber_wd(i) = eval_ber_local(ye_wd(idx_wd), idx_wd, tx_sym, NumPreamble, M);
    
    % RNN_AR
    [ye_ar, ~, idx_ar] = RNN_AR_Implementation(xRx, xTx, NumPreamble, 51, 32, 0.001, 20, 5, -20:20, [1 2]);
    ber_ar(i) = eval_ber_local(ye_ar(idx_ar), idx_ar, tx_sym, NumPreamble, M);
    
    disp(['SNR=' num2str(snr) ' | FFE=' num2str(ber_ffe(i)) ' | WD=' num2str(ber_wd(i)) ' | AR=' num2str(ber_ar(i))]);
end

%% Plot Results
figure('Name', 'RNN Crash Test: BER vs SNR');

% Handle BER=0 for log plot
min_val = 1e-6;
p_ffe = ber_ffe; p_ffe(p_ffe==0) = min_val;
p_wd  = ber_wd;  p_wd(p_wd==0)  = min_val;
p_ar  = ber_ar;  p_ar(p_ar==0)  = min_val;

semilogy(SNR_Sweep, p_ffe, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 7); hold on;
semilogy(SNR_Sweep, p_wd,  'k-s', 'LineWidth', 1.5, 'MarkerSize', 7, 'MarkerFaceColor', 'k');
semilogy(SNR_Sweep, p_ar,  'r-^', 'LineWidth', 1.5, 'MarkerSize', 7, 'MarkerFaceColor', 'r');

yline(3.8e-3, '--k', 'HD-FEC', 'LabelHorizontalAlignment', 'left');

grid on;
xlabel('SNR (dB)');
ylabel('BER (log scale)');
title({'RNN Robustness Test (Synthetic Channel)'; 'Strong ISI + Volterra Nonlinearity'});
legend('FFE (Linear)', 'RNN-WD (Hard FB)', 'RNN-AR (Soft FB)', 'Location', 'northeast');
set(gca, 'XDir', 'reverse'); % High SNR on left
ylim([min_val 1]);

% Add annotation for 0 BER
text(max(SNR_Sweep)-1, min_val*1.5, 'Floor: BER=0', 'FontSize', 8, 'Color', 'k');

% Print Channel Info
fprintf('\n--- Channel Model ---\n');
fprintf('ISI Coefficients: %s\n', mat2str(h_isi, 3));
fprintf('Nonlinearity: y = x + 0.1*x^2 - 0.05*x^3\n');

function ber = eval_ber_local(ye, idx, tx_sym, n_train, M)
    mask = (idx > n_train) & (idx <= length(tx_sym));
    if sum(mask) == 0, ber = 1; return; end
    y_test = ye(mask); x_test = tx_sym(idx(mask));
    scale = std(y_test) / std(pammod(x_test, M, 0, 'gray'));
    y_test = y_test / scale;
    y_sym = pamdemod(y_test, M, 0, 'gray');
    [~, ber] = biterr(y_sym(:), x_test(:), log2(M));
end