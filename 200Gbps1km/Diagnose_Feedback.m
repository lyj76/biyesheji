%% Feedback Diagnosis Script
clear;
close all;

%% add paths
addpath('fns');
addpath(fullfile('fns','fns2'));

%% 1. Load Data (rop3dBm)
load('rop3dBm_1.mat', 'ReData');
ReData = -ReData;

% Basic Params
M = 4;
NumSymbols = 2^18;
[xsym, xm] = PAMSource(M, NumSymbols);
xsym = xsym(:);
xs = xm;

% Sync & Filter
rolloff = 0.1; N = 128; Osamp_factor = 2;
h = fdesign.pulseshaping(Osamp_factor, 'Raised Cosine', 'N,Beta', N, rolloff);
Hd = design(h); sqrt_ht = Hd.Numerator; sqrt_ht = sqrt_ht ./ max(sqrt_ht);
x_upsamp = upsample(xs, Osamp_factor);
x_shape = conv(sqrt_ht, x_upsamp); x_shape = x_shape ./ sqrt(mean(abs(x_shape).^2));

[TE, ~] = TEFEMMM2(ReData, 1024, 0.3);
ysync = ReData(1024 + 20 + TE : end);
ysync = ysync(1 : length(x_shape));
yt_filter = ysync(1 + N/2 : length(ysync) - N/2);

xTx = xs;
xRx = yt_filter;
NumPreamble = 10000;

%% 2. Run Backbone (Linear FFE)
disp('--- Step 0: Backbone ---');
[~, ye_lin_raw] = FFE_2pscenter(xRx, xTx, NumPreamble, 111, 0.9999);
[off, d0] = align_offset_delay_by_ser(ye_lin_raw, xsym, NumPreamble, M, -60:60);
ye_lin = ye_lin_raw(off:2:end);
idxTx = (1:length(ye_lin))' + d0;

% Valid Mask
valid = idxTx >= 1 & idxTx <= length(xTx);
ye_lin = ye_lin(valid);
ye_lin = ye_lin(:); % Ensure col
xTx_valid = xTx(idxTx(valid));
scale = std(xTx_valid);
ye_lin_norm = ye_lin / scale;

%% 3. Manual Feedback Injection Test
disp('--- Step 1: Manual Parameter Injection Check ---');
Volterra_Tap = 5;
Rank = 2;
Input_Dim = Volterra_Tap + 2; % 5 + 2 Feedback

% Create Mock Params
P = zeros(Input_Dim, Rank);
Alpha = ones(Rank, 1);

% *** FORCE FEEDBACK WEIGHTS ***
% 设 P 的最后两行 (反馈部分) 为大数值
P(end-1:end, :) = 1.0; 
% 前面设为 0，这样输出应该完全由反馈决定

% Prepare Input (Test Phase Logic)
U_Test = zeros(100, Volterra_Tap); % Mock Input (Zeros)
Y_Lin_Test = zeros(100, 1);

% Run Inference Logic (extracted from Implementation_v2)
N_Total = 100;
y_res_total = zeros(N_Total, 1);
P_input = P(1:Volterra_Tap, :);
P_fb    = P(Volterra_Tap+1:end, :);

Z_input = U_Test * P_input; % Should be 0
fb_regs = [1; -1]; % Initial Mock Feedback (Non-Zero!)

disp(['   [Check] Initial Feedback Regs: ', num2str(fb_regs')]);
disp(['   [Check] Feedback Weights (P_fb):']);
disp(P_fb);

for i = 1:5
    z_fb = fb_regs' * P_fb; % [1 x R]
    z_curr = Z_input(i, :) + z_fb;
    res_val = (z_curr.^2) * Alpha;
    
    disp(['   Sample ', num2str(i), ' Output (y_res): ', num2str(res_val)]);
    
    % Update regs with dummy logic
    fb_regs(2) = fb_regs(1);
    fb_regs(1) = 0.5; % dummy
end

% Check 1: Did we get non-zero output?
if any(res_val ~= 0)
    disp('   >>> PASS: Inference Path is using Feedback Weights.');
else
    disp('   >>> FAIL: Inference Path ignored Feedback Weights.');
end

%% 4. Gradient Check
disp('--- Step 2: Gradient Flow Check ---');
% Mock Training Batch
BatchSize = 10;
U_Batch = randn(BatchSize, Volterra_Tap);
Target_Batch = randn(BatchSize, 1);
% Mock Feedback (Teacher Forcing)
Fb1 = randn(BatchSize, 1);
Fb2 = randn(BatchSize, 1);
U_Full = [U_Batch, Fb1, Fb2];

% Params
P = randn(Input_Dim, Rank);
Alpha = randn(Rank, 1);

% Forward
Z = U_Full * P;
y_res = (Z.^2) * Alpha;
err = y_res - Target_Batch;

% Backward (Log-Cosh)
grad_err = tanh(err);
grad_P = zeros(size(P));

for r = 1:Rank
    scale = grad_err .* (2 * Alpha(r) * Z(:, r));
    grad_P(:, r) = U_Full' * scale;
end
grad_P = grad_P / BatchSize;

% Check Gradient of Feedback Part (Last 2 rows)
Grad_Fb = grad_P(end-1:end, :);
disp('   [Check] Gradient of Feedback Weights:');
disp(Grad_Fb);

if any(abs(Grad_Fb(:)) > 1e-6)
    disp('   >>> PASS: Gradient is flowing to Feedback Weights.');
else
    disp('   >>> FAIL: Gradient Vanished for Feedback Weights.');
end

%% 5. Real Training Check
disp('--- Step 3: Real Training Dynamics ---');
% Run actual training for 5 epochs with Feedback enabled
[~, params_out, ~] = LowRank_Volterra_Implementation_v2(xRx, xTx, xsym, 10000, 2, [1e-2, 1e-3], 5, true);

P_learned = params_out.P;
P_fb_learned = P_learned(end-1:end, :);

disp('   [Check] Learned Feedback Weights (after 5 epochs):');
disp(P_fb_learned);

norm_input = norm(P_learned(1:end-2, :));
norm_fb = norm(P_fb_learned);
ratio = norm_fb / (norm_input + eps);

disp(['   Ratio of FB Weights / Input Weights: ', num2str(ratio)]);

if ratio < 1e-3
    disp('   >>> DIAGNOSIS: Feedback weights are NOT growing. (Physics or Initialization issue)');
else
    disp('   >>> DIAGNOSIS: Feedback weights ARE growing. (It is working)');
end
