%% compare_equalizers_performance.m (升级版)
% 这是一个专业的均衡器性能对比平台
% 支持多文件 (不同ROP) 批量测试
% 对比对象: FFE, VNLE, LE-DFE, DP-VFFE, CLUT-VDFE

clear;
close all;

%% 1. 系统参数设置
addpath('fns');
addpath(fullfile('fns','fns2'));

Ft = 200e9;                   
Osamp_factor = 2;             
NumSymbols = 2^17; 
NumPreamble = 0;              
M = 4;                        

% 随机数种子
s = RandStream.create('mt19937ar', 'seed', 529558);
prevStream = RandStream.setGlobalStream(s);

%% 2. 信号生成 (Reference)
disp('Generating Reference Signals...');
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

%% 3. 均衡器参数配置 (统一标准)
NumPreamble_TDE = 10000; 

% 通用参数
N1 = 111;  % FFE Linear
N2 = 21;   % FFE Volterra
D1 = 25;   % DFE Linear

% --- 关键调整: D2 设为 5 ---
D2 = 5;    % DFE Volterra (合理的短程非线性记忆)
WL = 1;    % FFE Order
WD = 1;    % DFE Order

% CLUT 参数
K_Lin = 18; 
K_Vol = 90; 
Lambda = 0.9999;

%% 4. 多文件测试循环
file_list = {'rop3dBm_1.mat', 'rop5dBm_1.mat'};
rop_labels = {'ROP 3dBm', 'ROP 5dBm'};

algo_names = {'Linear FFE', 'VNLE', 'LE-DFE', 'DP-VFFE', 'CLUT-VDFE'};
num_algos = length(algo_names);
num_files = length(file_list);

% 结果存储矩阵: [Algorithm x File]
Results_BER = zeros(num_algos, num_files);
Results_Time = zeros(num_algos, num_files);
All_MSE_Curves = cell(num_algos, num_files); % 存曲线数据

disp('==================================================');
disp('Starting Multi-ROP Performance Comparison...');
disp('==================================================');

for f_idx = 1 : num_files
    
    current_file = file_list{f_idx};
    disp(['>>> Processing File: ', current_file, ' (', rop_labels{f_idx}, ')']);
    
    % 加载与预处理
    load(current_file, 'ReData');
    ReData = -ReData; % 反相
    
    % 同步
    th = 0.3;
    [TE, FE] = TEFEMMM2(ReData, 1024, th);
    ysync = ReData(1024 + 20 + TE : end); 
    ysync = ysync(1 : length(x_shape));
    yt_filter = ysync(1 + N/2 : length(ysync) - N/2);
    
    xTx = xs;         
    xRx = yt_filter;  
    
    % 算法内循环
    for a_idx = 1 : num_algos
        name = algo_names{a_idx};
        disp(['    Running: ', name, ' ...']);
        
        tic; % 计时开始
        
        switch name
            case 'Linear FFE'
                [h, ye] = FFE_2pscenter(xRx, xTx, NumPreamble_TDE, N1, Lambda);
                
            case 'VNLE'
                [h, ye] = VNLE2_2pscenter(xRx, xTx, NumPreamble_TDE, N1, N2, Lambda, WL);
                
            case 'LE-DFE'
                % D2=0, WD=0 强制退化为线性 DFE
                [h, d, ye] = LE_FFE2ps_centerDFE_new(xRx, xTx, NumPreamble_TDE, N1, D1, Lambda, M, M/2);
                
            case 'DP-VFFE'
                 [h, d, ye] = DP_VFFE2pscenter_VDFE(xRx, xTx, NumPreamble_TDE, N1, N2, D1, D2, Lambda, WL, WD, M, M/2);
                
            case 'CLUT-VDFE'
                 [~, ye] = CLUT_VDFE_Implementation(xRx, xTx, NumPreamble_TDE, N1, N2, D1, D2, WL, WD, M, K_Lin, K_Vol, Lambda);
        end
        
        elapsed_time = toc;
        Results_Time(a_idx, f_idx) = elapsed_time;
        
        % 计算 BER
        calc_start = NumPreamble_TDE + 500;
        calc_end = length(ye);
        if calc_end > length(xsym), calc_end = length(xsym); end
        
        ym = Normalizepam(ye, M);
        ysym = dePAMSource(M, ym);
        
        range = calc_start : calc_end;
        [~, ber] = biterr(ysym(range), xsym(range), log2(M));
        Results_BER(a_idx, f_idx) = ber;
        
        % 计算 MSE 曲线
        err_curve = abs(ym(1:calc_end) - xs(1:calc_end)).^2;
        windowSize = 1000;
        err_smooth = filter(ones(1,windowSize)/windowSize, 1, err_curve);
        All_MSE_Curves{a_idx, f_idx} = 10*log10(err_smooth);
        
        disp(['       BER : ', num2str(ber, '%.2e')]);
    end
end

%% 5. 专业绘图展示

% 图 1: BER 对比 (分组柱状图)
figure('Position', [100, 100, 1000, 500]);
b = bar(log10(Results_BER)); % 分组: 算法为X轴，不同文件为不同颜色的柱子
xlabel('Algorithms');
ylabel('log10(BER)');
title(['BER Performance Comparison (D2 = ', num2str(D2), ')']);
set(gca, 'XTickLabel', algo_names);
legend(rop_labels, 'Location', 'Best');
grid on;
% 标注数值
for i = 1:num_algos
    for j = 1:num_files
        text(i + (j-1.5)*0.25, log10(Results_BER(i,j)), num2str(Results_BER(i,j), '%.1e'), ...
            'HorizontalAlignment','center', 'VerticalAlignment','bottom', 'FontSize', 8);
    end
end

% 图 2: 收敛曲线 (针对每个文件单独画图)
colors = {'b', 'g', 'm', 'r', 'k'};
line_styles = {'-', '-', '--', '-', '-'};
line_widths = [1, 1, 1, 1.5, 2];

for f_idx = 1 : num_files
    figure('Position', [100 + (f_idx-1)*50, 600 - (f_idx-1)*50, 800, 400]);
    hold on;
    for a_idx = 1 : num_algos
        plot(All_MSE_Curves{a_idx, f_idx}, 'Color', colors{a_idx}, 'LineStyle', line_styles{a_idx}, 'LineWidth', line_widths(a_idx));
    end
    hold off;
    grid on;
    xlabel('Symbol Index');
    ylabel('MSE (dB)');
    title(['MSE Convergence Curve - ', rop_labels{f_idx}]);
    legend(algo_names, 'Location', 'Best');
    ylim([-30, 10]);
end

disp('==================================================');
disp('Comparison Completed.');