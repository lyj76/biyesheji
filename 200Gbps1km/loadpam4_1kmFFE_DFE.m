%% pam (高速光通信 PAM4 信号处理主程序)
clear;
close all;

%% add paths (添加函数路径)
addpath('fns');
addpath(fullfile('fns','fns2'));

%% parameters (系统参数设置)
Ft = 200e9;                   % 采样率 200 GSa/s
Osamp_factor = 2;             % 过采样因子 (2 samples/symbol)
NumSymbols = 2^18;            % 符号总数
NumPreamble = 0;              % 前导码长度 (此处设为0，但在均衡器内部会切分训练集)
NumSym_total = NumSymbols + NumPreamble;
M = 4;                        % 调制阶数 (PAM-4)
 
% 随机数生成器 (保证结果可复现)
s = RandStream.create('mt19937ar', 'seed', 529558);
prevStream = RandStream.setGlobalStream(s);

%% MQAM modulate (PAM4 调制与信号生成)
[xsym, xm] = PAMSource(M, NumSymbols);
xs = xm; % xs 为发送的符号序列 (Ideal Symbols)
% xs(find(xs==3))=3;
% figure;plot(xs(:),'o');grid on;

%% pluse shaping (脉冲成型 - 升余弦滤波器)
rolloff = 0.1;
N = 128;
h = fdesign.pulseshaping(Osamp_factor, 'Raised Cosine', 'N,Beta', N, rolloff);
Hd = design(h);
sqrt_ht = Hd.Numerator;
sqrt_ht = sqrt_ht ./ max(sqrt_ht);

% 上采样并滤波 (模拟发送端 DAC)
x_upsamp = upsample(xs, Osamp_factor); % Upsampling
x_shape = conv(sqrt_ht, x_upsamp);
x_shape = x_shape ./ sqrt(mean(abs(x_shape).^2)); % Power Normalization

%% 数据加载与批处理循环
NN=101; % 默认使用 NN=101，恢复原始逻辑
% NN=11;
% NN=3:2:25;
% NN=11:10:201;

% 恢复被删除的 Ndata 注释块
% Numdata=15;
% Ndata=[7,10,13,1,12,14,6,9,4,11];%btb rop=5
% Ndata=[6,9,10,7,1,8,2,4,5,11];%1km rop=5

% 待处理的数据文件列表 (实验数据)
file_list = {'rop3dBm_1.mat', 'rop5dBm_1.mat'};
BERall = zeros(length(file_list), length(NN));

for n1 = 1 : length(file_list)
    % DSO160G 代码块
    if 0
        DSO160G
        ReData = resample(channel1data6(2,:), 100, 160);
        ReData = ReData - mean(ReData);
    end          

    disp(['Processing File Index: ', num2str(n1)]);
    load(file_list{n1}, 'ReData')

    % 信号反相 (实验常见操作，取决于PD极性)
    % 注意：原始代码中是在 load 之后、同步之前反相的，这里保持一致
    ReData = -ReData;
    
    %% synchronization (同步)
    th = 0.3;
    [TE, FE] = TEFEMMM2(ReData, 1024, th);
    
    % 手动微调同步位置
    abc = TE + 0;
    % abc = 100 + 0 + 0;
    TE = abc;
    
    % 截取同步后的数据
    ysync = ReData(1024 + 20 + TE : end); 
    % 长度对齐到发送数据长度 (x_shape)
    ysync = ysync(1 : length(x_shape));

    % 恢复原始的内层循环逻辑
    for m1 = 1 : length(NN)
    % for m1 = 1:1

        %% Match filtering (接收端匹配滤波)
        % 去除卷积带来的边缘效应
        yt_filter = ysync(1 + N/2 : length(ysync) - N/2);
        
        % 频谱分析 (调试用)
        % figure;pwelch(yt_filter(:),[],[],[],Ft,'twosided');

        %% RLS LE/NE (均衡器核心参数配置)
        
        xTx = xs;         % 发送端参考符号 (1 sps)
        xRx = yt_filter;  % 接收端信号 (2 sps)
        
        NumPreamble_TDE = 10000; % 用于训练的符号数 (Training Length)
        
        % --- 均衡器结构参数 (关键!) ---
        
        % FFE (前馈) 参数
        N1 = 111;  % FFE 线性部分长度 (Linear Memory)
        N2 = 21;   % FFE 非线性部分长度 (Nonlinear Memory)
        WL = 1;    % FFE 非线性窗口长度 (Lag)
                   % WL=1: 仅自平方 x(n)^2
                   % WL=3: 包含 x(n)x(n-1), x(n)x(n-2) 等
        
        % DFE (反馈) 参数
        D1 = 25;   % DFE 线性部分长度 (Linear Feedback Depth)
        
        % --- 关于 D2 的解释 ---
        % D2: DFE 非线性项覆盖范围 (Nonlinear Feedback Memory Depth)
        % 含义: 这是一个非常重要的参数，控制我们在反馈路径中考虑“多久以前”的非线性干扰。
        %       如果 D2=0，表示不启用非线性反馈。
        %       如果 D2=5，表示我们会用最近的 5 个判决符号来计算交叉项 (如 x(n-1)*x(n-2))。
        %       通常 D2 不需要设得像 D1 那么大，因为非线性记忆效应衰减很快。
        D2 = 0;   
        
        WD = 1;    % DFE 非线性窗口长度 (Lag)
        
        % CLUT-VDFE 聚类参数 (新增)
        K_Lin = 18;  % 线性簇数量
        K_Vol = 90;  % 非线性簇数量 (通常要大一些)

        % 缩放参数 (未使用/备用)
        aa = 0.3;
        bb = 1;

        %% 均衡器算法切换 (Algorithm Switching)
        % 请根据需要取消注释对应的行，一次只能运行一个算法
        
        % 1. 纯线性 FFE (Linear FFE)
        % [hffe,ye] = FFE_2pscenter(xRx,xTx,NumPreamble_TDE,N1,0.9999); 
        
        % 2. Volterra 非线性 FFE (VNLE)
        % [hffe,ye] = VNLE2_2pscenter(xRx,xTx,NumPreamble_TDE,N1,N2,0.9999,WL);
        
        % 3. 线性 FFE + 线性 DFE (Linear FFE + DFE)
        % [hffe,hdfe,ye] = LE_FFE2ps_centerDFE_new(xRx,xTx,NumPreamble_TDE,N1,D1,0.9999,M,M/2);  
        
        % 4. 全非线性 FFE + DFE (终极 BOSS)
         [hffe,hdfe,ye] = DP_VFFE2pscenter_VDFE(xRx,xTx,NumPreamble_TDE,N1,N2,D1,D2,0.9999,WL,WD,M,M/2);
        
        % 5. CLUT-VDFE (聚类查找表 VDFE) - 新增
        %[BER_clut, ye] = CLUT_VDFE_Implementation(xRx, xTx, NumPreamble_TDE, N1, N2, D1, D2, WL, WD, M, K_Lin, K_Vol, 0.9999);


        % 调试绘图 (恢复原有绘图代码)
        % figure;plot(hffe); hold on;plot(hdfe);
        
        %% Normalize (输出归一化与星座图显示)
        ym = Normalizepam(ye, M);
        
        % 恢复原本的绘图逻辑
        % eyediagram(ym(2000:3000),4) 
        % figure;hist(ym(:),1000);grid on;
        
        ytemp = ym(NumPreamble_TDE+1 : end); % 取出测试数据部分
        figure;plot(ytemp(:),'o');grid on;
        figure;hist(ytemp(:),1000);grid on;
        % hold on;hist(xm(1:3000),1000);grid on;

        %% MQAM demodulation (解调)
        ysym = dePAMSource(M, ym);

        %% BER/SNR Calculation (误码率与信噪比计算)
        % 跳过训练序列部分进行统计
        % 修正之前的语法错误，使用 length(ysym) 替代 end
        calc_range = NumPreamble_TDE+1 : length(ysym);
        
        % 注意：如果是 CLUT 算法，长度可能已经短了一截，所以这里要小心索引
        if length(calc_range) > length(ysym)
            calc_range = 1 : length(ysym); % Fallback
        end
        
        [ErrCount, BER_sub1] = biterr(ysym(calc_range), xsym(calc_range), log2(M));
        [ErrorSym, SER_sub1] = symerr(ysym(calc_range), xsym(calc_range));         
        
        % figure;plot(ysym(calc_range)- xsym(calc_range))
        
        [SNRdB_sub1, SNR1] = snr(xm(calc_range), ym(calc_range));

        %%%% 输出结果
        disp(['File ', num2str(n1), ' BER = ', num2str(BER_sub1)]);
        disp(['File ', num2str(n1), ' SNR (dB) = ', num2str(SNRdB_sub1)]);

        BERall(n1, m1) = BER_sub1;
    end
end

% 打印平均 BER
mean_ber = mean(BERall, 1);
disp(['Average BER = ', num2str(mean_ber)]);

% 排序与绘图代码块 (恢复原始逻辑)
if 0
    [aaa1, bbb1] = sort(BERall(:,end));
    avernber = mean(BERall(bbb1(1:10),:), 1);
    % figure;plot(NN,avernber)
end

% 最终绘图 (恢复)
berm = mean(BERall, 1);
figure;semilogy(NN, berm);
% figure;semilogy(NN,berm);
% berm
