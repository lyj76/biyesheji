%% RLS 算法验证与测试台 (System Identification Testbench)
% 作者: Gemini & User (Math PhD)
% 目的: 验证 FFE_2pscenter.m 的收敛性与准确性
% 原理: 构造已知信道 h_true，生成对应数据，看算法能否反解出 h_true

clear; close all; clc;

%% 1. 基础参数设置 (Configuration)
N1 = 21;                % 滤波器抽头长度 (Tap Length)
NumPreamble_TDE = 2000; % 训练数据长度 (数据量要足够让RLS收敛)
Lambda = 0.995;         % 遗忘因子 (接近1以保证稳态精度)

% 构造一个“上帝视角”的真实信道 (Ground Truth Channel)
% 我们用一个 Sinc 函数或者是高斯形状来模拟真实的物理信道响应
t_tap = linspace(-2, 2, N1);
h_true = sinc(t_tap); 
h_true = h_true(:) / sum(abs(h_true)); % 归一化，保持能量守恒

%% 2. 生成合成数据 (Synthetic Data Generation)
% Tx: 发送信号 (假设是 PAM-4 或 随机高斯白噪声)
% 注意：根据代码逻辑，Tx 的长度应该是 Rx 的 2 倍左右
L_Tx = NumPreamble_TDE * 2 + 100; 
xTx_raw = randn(L_Tx, 1) + 1i*randn(L_Tx, 1); % 复数信号

% 预处理：为了匹配函数内部的归一化逻辑，我们在外部先控制好幅度
xTx_raw = xTx_raw / mean(abs(xTx_raw));

%% 3. 模拟物理过程：通过信道 (Channel Simulation)
% 关键点：为了验证你的代码，我必须使用和你代码中 *完全一致* 的卷积/抽取逻辑
% 你的代码逻辑是：Rx(n) = h' * Tx_window(基于 2*n 索引)

L1 = (N1-1)/2;
xTx0_sim = [zeros(L1,1); xTx_raw; zeros(L1,1)]; % 保持和函数内一致的 padding
xRx_sim = zeros(NumPreamble_TDE, 1);

% --- 手动卷积 (Manual Convolution) ---
% 这就是“上帝”生成 Rx 的过程，没有任何噪声
for n = 1:NumPreamble_TDE
    % 提取和你代码一模一样的切片
    idx_center = 2*n; 
    % 注意：这里要极其小心索引对应。
    % 你的代码是：x = xTx0(2*n+(N1-1)/2+L1 : -1 : 2*n-(N1-1)/2+L1);
    % 为了严谨，我直接复制你的索引逻辑：
    indices = (2*n + (N1-1)/2 + L1) : -1 : (2*n - (N1-1)/2 + L1);
    
    % 容错处理：防止索引越界 (虽然理论上 padded 够长)
    if max(indices) > length(xTx0_sim) || min(indices) < 1
        continue;
    end
    
    x_vec = xTx0_sim(indices);
    
    % 生成无噪声 Rx
    xRx_sim(n) = h_true.' * x_vec;
end

% 添加噪声 (AWGN)
SNR_dB = 30; % 信噪比 30dB (比较干净，便于看收敛)
signal_power = mean(abs(xRx_sim).^2);
noise_power = signal_power / (10^(SNR_dB/10));
noise = sqrt(noise_power/2) * (randn(size(xRx_sim)) + 1i*randn(size(xRx_sim)));

xRx_input = xRx_sim + noise;

%% 4. 运行你的算法 (Run the Algorithm)
% 此时 xTx_raw 和 xRx_input 已经准备好，且数学关系明确
disp('正在运行 FFE_2pscenter...');
%[h_est, ye] = FFE_2pscenter(xTx_raw, xRx_input, NumPreamble_TDE, N1, Lambda);
[h_est, ye] = ffe(xTx_raw, xRx_input, NumPreamble_TDE, N1, Lambda);
%% 5. 结果验证与可视化 (Verification & Visualization)

% 5.1 误差收敛曲线 (Learning Curve)
error_signal = xRx_input(1:length(ye)) - ye(:);
mse_curve = 10*log10(abs(error_signal).^2);
% 平滑一下曲线以便观察
window_size = 50;
mse_smooth = filter(ones(1,window_size)/window_size, 1, mse_curve);

figure('Position', [100, 100, 1000, 600]);

subplot(2, 2, 1);
plot(mse_smooth, 'LineWidth', 1.5);
grid on;
title('MSE 收敛曲线 (Learning Curve)');
xlabel('迭代次数 (Iteration)');
ylabel('MSE (dB)');
legend('平滑后的误差');

% 5.2 滤波器系数对比 (Weights Comparison)
% 这是最硬核的检查：你算出来的 h 和我设定的 h_true 是否重合？
subplot(2, 2, 2);
stem(abs(h_true), 'o-', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', '真值 (Ground Truth)');
hold on;
stem(abs(h_est), 'x--', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', '估计值 (Estimated)');
grid on;
legend;
title('信道冲激响应对比 (Impulse Response)');
xlabel('抽头索引 (Tap Index)');
ylabel('幅度');

% 5.3 星座图/散点图 (Scatter Plot)
% 比较原始 Rx 和 均衡/拟合后的 ye
subplot(2, 2, 3);
plot(real(xRx_input), imag(xRx_input), '.', 'Color', [0.7 0.7 0.7], 'DisplayName', '带噪 Rx');
hold on;
plot(real(ye), imag(ye), 'r.', 'DisplayName', '拟合输出 Ye');
grid on;
axis equal;
title('信号拟合效果 (Rx Fitting)');
legend;

% 5.4 相位/复数域对比
subplot(2, 2, 4);
plot(real(h_true), imag(h_true), 'bo-', 'DisplayName', '真值');
hold on;
plot(real(h_est), imag(h_est), 'rx--', 'DisplayName', '估计值');
grid on;
axis equal;
title('滤波器复平面轨迹 (Complex Plane)');
legend;

disp('验证完成。请检查 Plot 2 中的 h 是否重合。');