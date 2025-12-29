function [h, ye] = VNLE2_2pscenter(xTx_in, xRx_in, NumPreamble_TDE, N1, N2, Lambda, WL)
% VNLE2_2pscenter: Volterra 非线性均衡器 (2倍过采样) - 健壮性修复版
%
% 输入:
%   xTx_in: 接收到的信号 (Input Rx), 2 samples/symbol
%   xRx_in: 期望的训练序列 (Desired Tx), 1 sample/symbol
%   NumPreamble_TDE: 训练长度

    %% 1. 初始化与预处理
    
    InputRx = xTx_in(:);
    DesiredTx = xRx_in(:);
    
    % 归一化
    InputRx = InputRx ./ mean(abs(InputRx(:)));
    DesiredTx = DesiredTx ./ mean(abs(DesiredTx(:)));
    
    % 截取训练数据
    InputRx_Train = InputRx(1 : NumPreamble_TDE * 2);
    DesiredTx_Train = DesiredTx(1 : NumPreamble_TDE);
    
    % 参数定义
    L_FFE_Lin = (N1 - 1) / 2;
    L_FFE_Vol = (N2 - 1) / 2;
    
    % 维度计算
    Dim_Vol = (2 * N2 - WL + 1) * WL / 2;
    NumCoeffs = N1 + Dim_Vol;
    
    %% 2. RLS 训练阶段
    
    % 补零 (Padding) - 关键修复：加足量的 Pad 防止越界
    PaddingLen = 1000;
    InputRx_Train_Pad = [zeros(PaddingLen, 1); InputRx_Train; zeros(L_FFE_Lin, 1)];
    DesiredTx_Train_Pad = [zeros(PaddingLen/2, 1); DesiredTx_Train];
    
    % RLS 初始化
    P = eye(NumCoeffs) * 0.01;
    h = zeros(NumCoeffs, 1);
    e = zeros(length(DesiredTx_Train_Pad), 1);
    
    % 循环起始点 (从 Pad 之后开始)
    StartIndex = PaddingLen/2 + 1;
    
    for n = StartIndex : length(DesiredTx_Train_Pad)
        
        loop_idx = n - StartIndex + 1; % 实际的符号索引 (1...NumPreamble)
        
        % 1. 线性项
        idx_lin_start = 2*n + L_FFE_Lin;
        idx_lin_end   = 2*n - L_FFE_Lin;
        x_Linear = InputRx_Train_Pad(idx_lin_start : -1 : idx_lin_end);
        
        % 2. 非线性项
        x_Nonlinear = [];
        for m = 1 : WL
            idx_nl_start = 2*n + L_FFE_Vol + (m-1);
            idx_nl_end   = 2*n - L_FFE_Vol + (m-1);
            
            vec_A = InputRx_Train_Pad(idx_nl_start : -1 : idx_nl_end);
            vec_B = InputRx_Train_Pad((idx_nl_start : -1 : idx_nl_end) - (m-1));
            x_Nonlinear = [x_Nonlinear; vec_A .* vec_B];
        end
        
        % 3. 拼接
        x_Total = [x_Linear; reshape(x_Nonlinear, [], 1)];
        
        % RLS 更新
        k = (P * x_Total) / (Lambda + x_Total.' * P * x_Total);
        y_curr = h.' * x_Total;
        e(loop_idx) = DesiredTx_Train_Pad(n) - y_curr;
        h = h + k * e(loop_idx);
        P = (P - k * (x_Total.' * P)) / Lambda;
    end
    
    disp(['VNLE Training MSE = ', num2str((abs(e(NumPreamble_TDE))).^2)]);
    
    %% 3. 应用阶段 (Application Phase)
    
    % 全量 Pad
    InputRx_Full_Pad = [zeros(PaddingLen, 1); InputRx; zeros(L_FFE_Lin, 1)];
    
    NumSyms = floor(length(InputRx) / 2);
    ye = zeros(1, NumSyms);
    
    for n = 1 : NumSyms
        
        % 这里的 n 是从 1 开始的，我们需要加上 Pad 的偏移才能对齐训练时的逻辑
        % 训练时 n 对应 StartIndex (即 501)。InputRx_Train_Pad(501) 是第一个有效数据
        % 这里 InputRx_Full_Pad(501) 也是第一个有效数据
        
        real_n = n + PaddingLen/2;
        
        % 1. 线性项
        idx_lin_start = 2*real_n + L_FFE_Lin;
        idx_lin_end   = 2*real_n - L_FFE_Lin;
        
        if idx_lin_start > length(InputRx_Full_Pad), break; end
        
        x_Linear = InputRx_Full_Pad(idx_lin_start : -1 : idx_lin_end);
        
        % 2. 非线性项
        x_Nonlinear = [];
        for m = 1 : WL
            idx_nl_start = 2*real_n + L_FFE_Vol + (m-1);
            idx_nl_end   = 2*real_n - L_FFE_Vol + (m-1);
            
            vec_A = InputRx_Full_Pad(idx_nl_start : -1 : idx_nl_end);
            vec_B = InputRx_Full_Pad((idx_nl_start : -1 : idx_nl_end) - (m-1));
            x_Nonlinear = [x_Nonlinear; vec_A .* vec_B];
        end
        
        x_Total = [x_Linear; reshape(x_Nonlinear, [], 1)];
        ye(n) = h.' * x_Total;
    end
    
    ye = ye(:).';

end