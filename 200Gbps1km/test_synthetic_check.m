%% Synthetic Forensic Test
clear;
close all;
addpath('fns');
addpath(fullfile('fns','fns2'));

%% 1. Simulation Parameters
NumSym = 20000;
M = 4;
NumPreamble = 5000; 
rng(123); 

%% 2. Generate Source
tx_sym = randi([0 M-1], NumSym, 1);
tx_amp = pammod(tx_sym, M, 0, 'gray');
tx_amp = tx_amp / std(tx_amp);
tx_upsamp = upsample(tx_amp, 2);

%% 3. Channel Model
h_isi = [0.05, 0.2, 0.5, 1.0, 0.5, 0.2, 0.05]; 
h_isi = h_isi / norm(h_isi);
rx_lin = conv(tx_upsamp, h_isi, 'same');
rx_nl = rx_lin + 0.1 * (rx_lin.^2) - 0.05 * (rx_lin.^3);

snr = 12;
disp(['--- DEEP FORENSIC at SNR = ' num2str(snr) ' dB ---']);
rx_noisy = awgn(rx_nl, snr, 'measured');

xRx = rx_noisy; 
xTx = tx_amp;   

%% --- Algo 2: RNN_WD ---
[ye_wd, ~, idx_wd] = RNN_WD_Implementation(xRx, xTx, NumPreamble, 51, 32, 0.001, 20, 10, -20:20, [1 2]);
st_wd = eval_ber_custom(ye_wd(idx_wd), idx_wd, tx_sym, NumPreamble, M);
disp(['RNN_WD BER: ' num2str(st_wd.BER)]);

%% --- Algo 3: RNN_AR ---
[ye_ar, ~, idx_ar] = RNN_AR_Implementation(xRx, xTx, NumPreamble, 51, 32, 0.001, 20, 5, -20:20, [1 2]);
st_ar = eval_ber_custom(ye_ar(idx_ar), idx_ar, tx_sym, NumPreamble, M);
disp(['RNN_AR BER: ' num2str(st_ar.BER)]);

%% --- FORENSIC ANALYSIS ---
disp('>>> FORENSIC ANALYSIS <<<');

test_mask = idx_ar > NumPreamble;
y_test = ye_ar(idx_ar(test_mask));
x_test = xTx(idx_ar(test_mask));

% 1. Variance of Error
err = y_test - x_test;
err_var = var(err);
disp(['RNN_AR Prediction Error Variance: ' num2str(err_var)]);

% 2. Correlation
corr_val = corr(y_test, x_test);
disp(['RNN_AR Correlation with Truth:    ' num2str(corr_val)]);

% 3. Check for Identity
if err_var < 1e-6
    disp('!!! ALERT: LEAKAGE CONFIRMED (Variance too low) !!!');
else
    disp('    STATUS: Looks legitimate (Variance > 1e-6).');
end

%% Helper
function stats = eval_ber_custom(ye, idx, tx_sym, n_train, M)
    mask = (idx > n_train) & (idx <= length(tx_sym));
    if sum(mask) == 0, stats.BER = NaN; return; end
    y_test = ye(mask);
    x_test = tx_sym(idx(mask));
    scale = std(y_test) / std(pammod(x_test, M, 0, 'gray'));
    y_test = y_test / scale;
    y_sym = pamdemod(y_test, M, 0, 'gray');
    y_sym = y_sym(:); x_test = x_test(:);
    [~, ber] = biterr(y_sym, x_test, log2(M));
    stats.BER = ber;
end
