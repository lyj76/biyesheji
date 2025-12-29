%% compare_equalizers_performance.m
% 这是一个专业的均衡器性能对比平台
% 对比对象: FFE, VNLE, LE-DFE, DP-VFFE, CLUT-VDFE
% 指标: BER, SER, 运行时间, MSE收敛曲线

clear;
close all;

%% 1. 系统参数设置
addpath('fns');
addpath(fullfile('fns','fns2'));

Ft = 200e9;                   
Osamp_factor = 2;             
NumSymbols = 2^17; % 稍微减小一点点以加快对比速度 (原 2^18)
NumPreamble = 0;              
M = 4;                        

% 随机数种子
s = RandStream.create('mt19937ar', 'seed', 529558);
prevStream = RandStream.setGlobalStream(s);

%% 2. 信号生成与加载
disp('Generating Signals...');
[xsym, xm] = PAMSource(M, NumSymbols);
xs = xm;

rolloff = 0.1;
N = 128;
h = fdesign.pulseshaping(Osamp_factor, 'Raised Cosine', 'N,Beta', N, rolloff);
Hd = design(h);
sqrt_ht = Hd.Numerator;
sqrt_ht = sqrt_ht ./ max(sqrt_ht);

x_upsamp = upsample(xs, Osamp_factor);
x_shape = conv(sqrt_ht, x_upsamp);
x_shape = x_shape ./ sqrt(mean(abs(x_shape).^2));

% 加载实验数据 (只取第一个文件演示)
file_name = 'rop3dBm_1.mat';
disp(['Loading Data: ', file_name]);
load(file_name, 'ReData');
ReData = -ReData; % 反相

% 同步
th = 0.3;
[TE, FE] = TEFEMMM2(ReData, 1024, th);
ysync = ReData(1024 + 20 + TE : end); 
ysync = ysync(1 : length(x_shape));
yt_filter = ysync(1 + N/2 : length(ysync) - N/2);

xTx = xs;         
xRx = yt_filter;  

%% 3. 均衡器参数统一配置
NumPreamble_TDE = 10000; % 训练长度

% 通用参数
N1 = 111;  % Linear FFE Mem
N2 = 21;   % Volterra FFE Mem
D1 = 25;   % Linear DFE Mem
D2 = 0;    % Volterra DFE Mem (设为0以公平对比 CLUT-VDFE 的线性反馈部分，或设小值)
WL = 1;    % FFE Vol order
WD = 1;    % DFE Vol order

K_Lin = 18; % CLUT
K_Vol = 90; % CLUT
Lambda = 0.9999;

%% 4. 算法性能测试循环
% 定义算法列表
algo_names = {'Linear FFE', 'VNLE', 'LE-DFE', 'DP-VFFE', 'CLUT-VDFE'};
num_algos = length(algo_names);

results_BER = zeros(1, num_algos);
results_Time = zeros(1, num_algos);
mse_curves = {}; % 存储每个算法的 MSE 曲线 (如果有)

disp('--------------------------------------------------');
disp('Starting Performance Comparison...');
disp('--------------------------------------------------');

for i = 1 : num_algos
    name = algo_names{i};
    disp(['Running Algorithm: ', name, ' ...']);
    
    tic; % Start Timer
    
    switch name
        case 'Linear FFE'
            [h, ye] = FFE_2pscenter(xRx, xTx, NumPreamble_TDE, N1, Lambda);
            
        case 'VNLE'
            [h, ye] = VNLE2_2pscenter(xRx, xTx, NumPreamble_TDE, N1, N2, Lambda, WL);
            
        case 'LE-DFE'
            % D2=0, WD=0 模拟纯线性 DFE
            [h, d, ye] = LE_FFE2ps_centerDFE_new(xRx, xTx, NumPreamble_TDE, N1, D1, Lambda, M, M/2);
            
        case 'DP-VFFE'
             [h, d, ye] = DP_VFFE2pscenter_VDFE(xRx, xTx, NumPreamble_TDE, N1, N2, D1, D2, Lambda, WL, WD, M, M/2);
            
        case 'CLUT-VDFE'
             [~, ye] = CLUT_VDFE_Implementation(xRx, xTx, NumPreamble_TDE, N1, N2, D1, D2, WL, WD, M, K_Lin, K_Vol, Lambda);
    end
    
    elapsed_time = toc; % Stop Timer
    results_Time(i) = elapsed_time;
    
    % 计算 BER
    % 统一数据长度处理
    calc_start = NumPreamble_TDE + 500; % 避开训练和切换瞬态
    calc_end = length(ye);
    if calc_end > length(xsym), calc_end = length(xsym); end
    
    ym = Normalizepam(ye, M);
    ysym = dePAMSource(M, ym);
    
    range = calc_start : calc_end;
    [~, ber] = biterr(ysym(range), xsym(range), log2(M));
    results_BER(i) = ber;
    
    % 估算 MSE 曲线 (Error between ye and ideal symbols)
    % 注意：这只是为了绘图展示收敛趋势，并非严格的 RLS 内部 MSE
    % 我们计算 |ye_normalized - x_ideal|^2
    err_curve = abs(ym(1:calc_end) - xs(1:calc_end)).^2;
    % 做一个滑动平均让曲线平滑一点
    windowSize = 1000;
    err_smooth = filter(ones(1,windowSize)/windowSize, 1, err_curve);
    mse_curves{i} = 10*log10(err_smooth);
    
    disp(['   -> Time: ', num2str(elapsed_time, '%.4f'), ' s']);
    disp(['   -> BER : ', num2str(ber, '%.2e')]);
end

%% 5. 专业绘图

% Figure 1: BER Comparison (Bar Chart)
figure('Position', [100, 100, 800, 400]);
subplot(1, 2, 1);
bar(categorical(algo_names), log10(results_BER));
ylabel('log10(BER)');
title('BER Performance (Lower is Better)');
grid on;
% 在柱子上标注数值
text(1:num_algos, log10(results_BER), num2str(results_BER', '%.1e'), ...
    'HorizontalAlignment','center', 'VerticalAlignment','bottom');

% Figure 2: Execution Time Comparison
subplot(1, 2, 2);
bar(categorical(algo_names), results_Time);
ylabel('Execution Time (s)');
title('Computational Complexity (Lower is Better)');
grid on;
% 标注数值
text(1:num_algos, results_Time, num2str(results_Time', '%.2f'), ...
    'HorizontalAlignment','center', 'VerticalAlignment','bottom');

% Figure 3: Convergence (MSE Learning Curve)
figure('Position', [100, 600, 800, 500]);
hold on;
colors = {'b', 'g', 'm', 'r', 'k'};
line_styles = {'-', '-', '--', '-', '-'};
line_width = [1, 1, 1, 1.5, 2];

for i = 1 : num_algos
    plot(mse_curves{i}, 'Color', colors{i}, 'LineStyle', line_styles{i}, 'LineWidth', line_width{i});
end
hold off;
grid on;
xlabel('Symbol Index');
ylabel('MSE (dB)');
title('MSE Convergence Analysis');
legend(algo_names, 'Location', 'Best');
ylim([-30, 10]); % 根据实际情况调整视野

disp('--------------------------------------------------');
disp('Comparison Completed.');
