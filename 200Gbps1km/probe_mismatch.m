%% Probe Script: Check Mismatch between Generated Tx and Loaded Rx
clear; close all;
addpath('fns'); % Add function path

%% 1. Load Rx Data
data_path = 'C:\Users\27456\Desktop\毕业设计\PS_PAM4_2_RoF_0.1_Cal_45GHz_AWG_210mV_ROP_5.6dBm.mat';
disp(['Loading Rx data from: ', data_path]);
load(data_path, 'rx_original');
Rx_segment = rx_original(1, :); % Take the first segment
Rx_segment = Rx_segment(:) - mean(Rx_segment); % Remove DC

%% 2. Generate Tx Data (using PAMSource + Seed)
M = 4;
NumSymbols = 2^17; 
s = RandStream.create('mt19937ar', 'seed', 529551);
RandStream.setGlobalStream(s);

disp('Generating Tx using PAMSource (Uniform) - TRYING BINARY MAPPING...');
% [xsym, xm] = PAMSource(M, NumSymbols); % This was Gray
% Manual generation for Binary mapping check
Tx_sym_idx = randi([0 M-1], NumSymbols, 1);
Tx_seq = pammod(Tx_sym_idx, M, 0, 'bin'); % Try Binary Mapping

%% 3. Pre-process Rx for Correlation
Osamp = 2;
Tx_upsampled = rectpulse(Tx_seq, Osamp); 

% Crop for speed
L_corr = 10000; 
Tx_probe = Tx_upsampled(1:L_corr);
Rx_probe = Rx_segment(10000 : 10000 + L_corr - 1);

%% 4. Cross Correlation
disp('Calculating Cross-Correlation (Raw & Abs)...');
% Check Raw
[c, lags] = xcorr(Rx_probe, Tx_probe, 'coeff'); 
[max_c, idx] = max(c);
[min_c, idx_min] = min(c);

% Check Abs (to confirm envelope match persists)
[c_abs, ~] = xcorr(abs(Rx_probe), abs(Tx_probe), 'coeff');
max_c_abs = max(c_abs);

disp('------------------------------------------------');
disp(['Mapping: BINARY']);
disp(['Max Correlation (Raw): ', num2str(max_c)]);
disp(['Min Correlation (Raw): ', num2str(min_c)]);
disp(['Max Correlation (Abs): ', num2str(max_c_abs)]);
disp('------------------------------------------------');


%% 5. Visual Comparison (Blind Recovery attempt)
% Let's try to look at the first few symbols
disp('------------------------------------------------');
disp('Visual Check (First 20 symbols):');
% Normalize Rx crudely
Rx_norm = Rx_segment / std(Rx_segment);
% Hard limit to -3, -1, 1, 3 roughly
% This is just for visual inspection
disp('Generated Tx (First 10):');
disp(Tx_seq(1:10).');

% Since we don't know the alignment, we can't print the "corresponding" Rx.
% But the low correlation is usually enough proof.
