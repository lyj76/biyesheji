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

function ber = eval_ber_local(ye, idx, tx_sym, n_train, M)
    mask = (idx > n_train) & (idx <= length(tx_sym));
    if sum(mask) == 0, ber = 1; return; end
    y_test = ye(mask); x_test = tx_sym(idx(mask));
    scale = std(y_test) / std(pammod(x_test, M, 0, 'gray'));
    y_test = y_test / scale;
    y_sym = pamdemod(y_test, M, 0, 'gray');
    [~, ber] = biterr(y_sym(:), x_test(:), log2(M));
end