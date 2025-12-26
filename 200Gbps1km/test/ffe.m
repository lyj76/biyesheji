function [h, ye] = ffe(xTx, xRx, NumPreamble_TDE, N1, Lambda)
% FFE: 完全保留原始逻辑的封装版本
% 直接搬运自你提供的 FFE_2pscenter 代码

    %% 1. 数据对齐与归一化 (完全保留原始逻辑)
    xTx = xTx(:);
    xRx = xRx(:);
    
    % --- 原始代码的归一化 (Magic Scaling) ---
    % 这一步非常关键，RLS 的 P 矩阵更新极度依赖信号幅度
    xTx = xTx ./ mean(abs(xTx(:)));
    xRx = xRx ./ mean(abs(xRx(:)));
    
    Rxdata = xRx;
    Txdata = xTx;
    
    % 截取训练数据
    xTx_train = xTx(1:NumPreamble_TDE*2); 
    xRx_train = xRx(1:NumPreamble_TDE);
    
    L1 = (N1-1)/2;
    
    %% 2. 第一阶段：RLS 迭代训练 h (Training Phase)
    xTx0 = xTx_train(:);
    xRx0 = xRx_train(:);
    y = zeros(size(xRx0));
    
    % --- 原始初始化 (Magic Initialization) ---
    % 你原来用的是 0.01，这对应很强的正则化/低增益
    P = eye(N1) * 0.01; 
    
    h = zeros(N1, 1); % 确保是列向量
    
    % Padding
    xTx0 = [zeros(L1,1); xTx0; zeros(L1,1)];
    
    for n = 1:length(xRx0)
        % 原始索引逻辑
        % x=xTx0(2*n+(N1-1)/2+L1:-1:2*n-(N1-1)/2+L1);
        % 为了代码稳健性，我只把索引计算拆开写，逻辑不变
        idx_start = 2*n + (N1-1)/2 + L1;
        idx_end   = 2*n - (N1-1)/2 + L1;
        
        x = xTx0(idx_start : -1 : idx_end);
        
        % RLS Update
        k = P * x ./ (Lambda + x.' * P * x);
        y(n) = h.' * x;
        e_n = xRx0(n) - y(n);
        h = h + k * e_n;
        P = (P - k * (x.' * P)) / Lambda;
    end
    
    %% 3. 第二阶段：应用滤波器生成 ye (Application Phase)
    % 原始代码逻辑：用算好的 h，跑一遍全量数据 Txdata
    
    xTx1 = [zeros(L1,1); Txdata; zeros(L1,1)];
    ye = zeros(size(Rxdata));
    
    for n = 1:length(Rxdata)
        % 同样的索引逻辑
        idx_start = 2*n + (N1-1)/2 + L1;
        idx_end   = 2*n - (N1-1)/2 + L1;
        
        % 防止在全量数据边缘越界 (加一个简单的保护)
        if idx_start > length(xTx1)
             break; 
        end
        
        x = xTx1(idx_start : -1 : idx_end);
        ye(n) = h.' * x;
    end
    
    % 转置输出以匹配原有格式
    ye = ye(:).'; 
    h = h(:);
end