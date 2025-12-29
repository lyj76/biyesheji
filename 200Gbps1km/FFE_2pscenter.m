function [h, ye] = FFE_2pscenter(xTx_in, xRx_in, NumPreamble_TDE, N1, Lambda)
% FFE_2pscenter: 分数间隔前馈均衡器 (2倍过采样输入 -> 1倍符号率输出)
%
% 本函数使用 递归最小二乘法 (RLS) 实现自适应均衡。
% 适用于输入信号为 2 samples/symbol，输出为 1 sample/symbol 的系统。
%
% 关于原代码变量名的修正说明：
%   原函数定义为：[h, ye] = FFE_2pscenter(xTx, xRx, ...)
%   但在主脚本调用时却是：FFE_2pscenter(ReData, xs, ...)
%   这意味着函数内部的 'xTx' 实际上接收的是【接收端数据(Rx)】，
%   而 'xRx' 接收的是【发送端参考数据(Tx)】。
%   
%   为了清晰起见，下方代码已重命名变量：
%   InputRx   <- xTx_in (接收到的信号，高采样率)
%   DesiredTx <- xRx_in (期望信号/参考信号，符号率)

    %% 1. 初始化与预处理
    
    % 确保输入为列向量
    InputRx = xTx_in(:);   
    DesiredTx = xRx_in(:); 
    
    % 归一化 (均值幅度归一化)
    % 将信号缩放，使其平均幅度为 1，有助于算法数值稳定性
    InputRx = InputRx ./ mean(abs(InputRx(:)));
    DesiredTx = DesiredTx ./ mean(abs(DesiredTx(:)));
    
    % 保存完整的接收信号，用于最后的全量滤波阶段
    FullInputRx = InputRx;
    
    % 截取用于“训练”的数据段
    % 输入是 2倍过采样，所以长度取 2 * NumPreamble_TDE
    InputRx_Train = InputRx(1 : NumPreamble_TDE * 2); 
    DesiredTx_Train = DesiredTx(1 : NumPreamble_TDE);
    
    FilterLen = N1;              % 均衡器滤波器的抽头长度 (Tap Length)
    HalfFilterLen = (FilterLen - 1) / 2; % 半长度，用于确定中心位置
    Padding = HalfFilterLen;     % 补零长度，处理边缘效应
    
    %% 2. RLS 训练阶段 (自适应学习滤波器系数 h)
    
    % 对输入训练信号的首尾进行补零，防止索引越界
    InputRx_Train_Pad = [zeros(Padding, 1); InputRx_Train; zeros(Padding, 1)];
    
    % 初始化 RLS 算法所需的变量
    % P: 逆相关矩阵，初始化为一个大的单位矩阵 (0.01 * Eye)
    P = eye(FilterLen) * 0.01; 
    % h: 滤波器系数 (权重)，初始化为全零
    h = zeros(FilterLen, 1); 
    
    % 误差向量，用于记录每次迭代的误差
    e = zeros(length(DesiredTx_Train), 1); 
    
    % --- RLS 迭代循环 ---
    for n = 1:length(DesiredTx_Train)
        % --- 步骤 A: 构建回归向量 (滑动窗口) ---
        % 从接收信号中提取一段长度为 FilterLen 的数据。
        % 索引 '2*n' 是因为输入是 2 samples/symbol (下采样逻辑)。
        % 窗口以当前时刻为中心。
        % 加上 'Padding' 是因为 InputRx_Train_Pad 前面补了零。
        
        idx_start = 2*n + HalfFilterLen + Padding;
        idx_end   = 2*n - HalfFilterLen + Padding;
        
        % 注意：这里索引是递减的 (-1)，这是为了配合矩阵乘法模拟卷积操作
        % 对应数学公式中的向量 x(n)
        x_vec = InputRx_Train_Pad(idx_start : -1 : idx_end);
        
        % --- 步骤 B: RLS 核心算法 ---
        
        % 1. 计算卡尔曼增益 (Kalman Gain, k)
        %    衡量当前输入对更新权重的影响程度
        %    公式: k = (P * x) / (Lambda + x' * P * x)
        k = (P * x_vec) / (Lambda + x_vec.' * P * x_vec);
        
        % 2. 滤波/预测 (Filtering)
        %    利用旧的系数计算当前输出
        %    公式: y = h' * x
        y_curr = h.' * x_vec;
        
        % 3. 计算误差 (Error Calculation)
        %    误差 = 期望值(参考信号) - 实际输出值
        e(n) = DesiredTx_Train(n) - y_curr;
        
        % 4. 更新滤波器系数 (Update Weights h)
        %    公式: h_new = h_old + k * error
        h = h + k * e(n);
        
        % 5. 更新逆相关矩阵 (Update P Matrix)
        %    公式: P_new = (P_old - k * x' * P_old) / Lambda
        P = (P - k * (x_vec.' * P)) / Lambda;
    end
    
    % 调试输出：训练阶段结束时的均方误差 (MSE)
    disp(['Final Training MSE (最终训练误差) = ', num2str((abs(e(end))).^2)]);
    
    %% 3. 应用阶段 (使用训练好的 h 对全量数据进行滤波)
    
    % 此时 h 已经收敛（理想情况下），我们将其应用到整个接收信号上
    
    % 同样对全量信号补零
    FullInputRx_Pad = [zeros(Padding, 1); FullInputRx; zeros(Padding, 1)];
    
    % 初始化输出向量
    % 输出符号数 = 输入采样点数 / 2
    NumTotalSymbols = floor(length(FullInputRx) / 2); 
    ye = zeros(1, NumTotalSymbols); 
    
    % --- 全量滤波循环 ---
    % 这里的逻辑与训练阶段的数据提取完全一致，确保相位对齐
    for n = 1:NumTotalSymbols
        
        % 构建滑动窗口
        idx_start = 2*n + HalfFilterLen + Padding;
        idx_end   = 2*n - HalfFilterLen + Padding;
        
        % 边界检查，防止处理到最后几个点时越界
        if idx_start > length(FullInputRx_Pad)
            break; 
        end
        
        x_vec = FullInputRx_Pad(idx_start : -1 : idx_end);
        
        % 应用滤波器 (点积)
        ye(n) = h.' * x_vec;
    end
    
    % 确保输出是行向量，方便后续处理
    ye = ye(:).';

end
