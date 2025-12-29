function [h, ye] = VNLE2_2pscenter(xTx_in, xRx_in, NumPreamble_TDE, N1, N2, Lambda, WL)
% VNLE2_2pscenter: Volterra 非线性均衡器 (2倍过采样)
%
% 核心思想：通过构建包含由输入信号的高阶项（如平方、交叉乘积）组成的“扩展向量”，
% 将非线性均衡问题转化为高维空间中的线性 RLS 问题。
%
% 输入:
%   xTx_in: 接收到的信号 (Input Rx), 2 samples/symbol
%   xRx_in: 期望的训练序列 (Desired Tx), 1 sample/symbol
%   N1:     线性滤波器的抽头长度 (Linear Tap Length)
%   N2:     非线性滤波器的覆盖范围 (Nonlinear Memory Depth)
%   WL:     非线性窗口长度 (Volterra Kernel Size/Lag), 控制考虑多少个交叉项
%
% 输出:
%   h:      训练好的复合系数向量 (包含线性部分和非线性部分)
%   ye:     均衡后的输出信号

    %% 1. 初始化与预处理 (Initialization)
    
    % 确保输入为列向量
    InputRx = xTx_in(:);    % 高采样率接收信号
    DesiredTx = xRx_in(:);  % 符号率期望信号
    
    % 归一化 (均值幅度)
    InputRx = InputRx ./ mean(abs(InputRx(:)));
    DesiredTx = DesiredTx ./ mean(abs(DesiredTx(:)));
    
    % 备份全量数据用于应用阶段
    InputRx_Full = InputRx;
    % DesiredTx_Full = DesiredTx; % 未使用
    
    % 截取训练数据
    InputRx_Train = InputRx(1 : NumPreamble_TDE * 2);
    DesiredTx_Train = DesiredTx(1 : NumPreamble_TDE);
    
    % 参数定义
    HalfLen_Linear = (N1 - 1) / 2;     % 线性部分中心偏移
    HalfLen_Nonlinear = (N2 - 1) / 2;  % 非线性部分中心偏移
    
    % 计算系数总长度 (Total Coefficients)
    % 线性部分长度: N1
    % 非线性部分长度: 这是一个等差数列求和相关的逻辑
    % 具体的交叉项数量取决于 N2 和 WL
    NumCoeffs = N1 + (2 * N2 - WL + 1) * WL / 2;
    
    %% 2. RLS 训练阶段 (Training Phase)
    
    % 补零以处理边界 (Padding)
    % 在头部和尾部补零，长度取线性部分的半长
    InputRx_Train_Pad = [zeros(HalfLen_Linear, 1); InputRx_Train; zeros(HalfLen_Linear, 1)];
    DesiredTx_Train = DesiredTx_Train; % 保持不变
    
    % RLS 初始化
    P = eye(NumCoeffs) * 0.01;  % 逆相关矩阵
    h = zeros(NumCoeffs, 1);    % 复合系数向量
    e = zeros(length(DesiredTx_Train), 1); % 误差记录
    
    % --- RLS 迭代循环 ---
    for n = 1 : length(DesiredTx_Train)
        
        % --- A. 构建 Volterra 输入向量 (Feature Engineering) ---
        
        % 1. 线性项 (Linear Term): 标准的 FFE 窗口
        idx_lin_start = 2*n + HalfLen_Linear;
        idx_lin_end   = 2*n - HalfLen_Linear;
        x_Linear = InputRx_Train_Pad(idx_lin_start : -1 : idx_lin_end);
        
        % 2. 非线性项 (Volterra Kernels): 生成交叉乘积项
        x_Nonlinear = [];
        for m = 1 : WL
            % m 代表“延迟差” (Lag)。
            % 我们计算当前窗口信号 与 延迟 m 后的信号 的乘积。
            % 比如: x(k) * x(k) (m=1, 自平方)
            %       x(k) * x(k-1) (m=2, 相邻交叉)
            
            % 提取两个用于相乘的子向量
            % 范围基于 N2 (非线性覆盖范围)
            % 这里的索引逻辑非常复杂，是为了在下采样(2ps)的同时对齐相位
            
            % 向量 A: 基准向量
            idx_nl_start = 2*n + HalfLen_Nonlinear + (m-1); % 这里的偏移逻辑保留原代码
            idx_nl_end   = 2*n - HalfLen_Nonlinear + (m-1);
            vec_A = InputRx_Train_Pad(idx_nl_start : -1 : idx_nl_end);
            
            % 向量 B: 延迟 m 后的向量 (实际上是上述索引再减去 m-1)
            % 实际上 vec_B 就是 vec_A 向右平移了 m-1 个单位
            vec_B = InputRx_Train_Pad((idx_nl_start : -1 : idx_nl_end) - (m-1));
            
            % 逐元素相乘得到二阶项
            x_Nonlinear = [x_Nonlinear; vec_A .* vec_B];
        end
        
        % 3. 拼接最终向量 (Regression Vector)
        % x_Total = [ 线性项; 非线性项 ]
        x_Total = [x_Linear; reshape(x_Nonlinear, [], 1)];
        
        % --- B. RLS 核心更新 ---
        
        % 计算增益 k
        % 分母是一个标量: Lambda + x'Px
        k = (P * x_Total) / (Lambda + x_Total.' * P * x_Total);
        
        % 预测输出 y
        y_curr = h.' * x_Total;
        
        % 计算误差 e
        e(n) = DesiredTx_Train(n) - y_curr;
        
        % 更新系数 h
        h = h + k * e(n);
        
        % 更新 P 矩阵
        P = (P - k * (x_Total.' * P)) / Lambda;
    end
    
    disp(['VNLE Training MSE = ', num2str((abs(e(end))).^2)]);
    
    %% 3. 应用阶段 (Application Phase)
    
    % 准备全量输入数据 (同样需要补零)
    InputRx_Full_Pad = [zeros(HalfLen_Linear, 1); InputRx_Full; zeros(HalfLen_Linear, 1)];
    
    % 输出长度 (下采样)
    NumSyms = floor(length(InputRx_Full) / 2);
    ye = zeros(1, NumSyms);
    
    % --- 全量滤波循环 ---
    for n = 1 : NumSyms
        
        % --- A. 构建 Volterra 输入向量 (逻辑同上) ---
        
        % 1. 线性项
        idx_lin_start = 2*n + HalfLen_Linear;
        idx_lin_end   = 2*n - HalfLen_Linear;
        
        if idx_lin_start > length(InputRx_Full_Pad)
            break; 
        end
        
        x_Linear = InputRx_Full_Pad(idx_lin_start : -1 : idx_lin_end);
        
        % 2. 非线性项
        x_Nonlinear = [];
        for m = 1 : WL
            idx_nl_start = 2*n + HalfLen_Nonlinear + (m-1);
            idx_nl_end   = 2*n - HalfLen_Nonlinear + (m-1);
            
            % 边界检查 (可选，取决于 Pad 是否足够大，这里假设足够)
            if idx_nl_start > length(InputRx_Full_Pad)
               % 简单的边界处理，防止 crash
               vec_A = zeros(N2,1); vec_B = zeros(N2,1); 
            else
               vec_A = InputRx_Full_Pad(idx_nl_start : -1 : idx_nl_end);
               vec_B = InputRx_Full_Pad((idx_nl_start : -1 : idx_nl_end) - (m-1));
            end
            
            x_Nonlinear = [x_Nonlinear; vec_A .* vec_B];
        end
        
        % 3. 拼接
        x_Total = [x_Linear; reshape(x_Nonlinear, [], 1)];
        
        % --- B. 滤波 ---
        ye(n) = h.' * x_Total;
    end
    
    % 调整输出格式
    ye = ye(:).';

end


