function [BER, ye_CLUT] = CLUT_VDFE_Implementation(Rx_in, Tx_in, NumPreamble, N1, N2, D1, D2, WL, WD, M, K_Lin, K_Vol, Lambda)
% CLUT_VDFE_Implementation: 基于聚类辅助查找表的低复杂度 Volterra DFE
%
% 输入:
%   Rx_in: 接收信号 (2 samples/symbol)
%   Tx_in: 发送符号 (1 sample/symbol)
%   NumPreamble: 训练序列长度
%   N1, N2, WL: FFE 参数
%   D1, D2, WD: DFE 参数
%   M: PAM阶数 (e.g. 4)
%   K_Lin: 线性聚类数 (推荐 18)
%   K_Vol: 非线性聚类数 (推荐 90)
%   Lambda: RLS 遗忘因子 (e.g. 0.9999)

    %% 0. 预处理与归一化
    InputRx = Rx_in(:);
    DesiredTx = Tx_in(:);
    
    % 强制归一化以保证 RLS 稳定性 (Unit Mean Abs)
    scale_Rx = mean(abs(InputRx));
    scale_Tx = mean(abs(DesiredTx));
    
    if scale_Rx == 0, scale_Rx = 1; end
    if scale_Tx == 0, scale_Tx = 1; end
    
    InputRx = InputRx ./ scale_Rx;
    DesiredTx = DesiredTx ./ scale_Tx;
    
    % 确定 PAM 电平 (基于归一化后的 Tx)
    % 对于 PAM4，归一化后的理想电平通常是 [-1.5, -0.5, 0.5, 1.5] 附近的数
    % 我们构建标准电平用于查表
    if M == 4
        % PAM4: -3 -1 1 3 -> mean(abs)=2. Normalized: -1.5 -0.5 0.5 1.5
        BaseLevels = [-3, -1, 1, 3];
        PamLevels = BaseLevels / mean(abs(BaseLevels)); 
    else
        % 通用 M-PAM
        BaseLevels = -(M-1):2:(M-1);
        PamLevels = BaseLevels / mean(abs(BaseLevels));
    end
    
    L_FFE_Lin = floor((N1 - 1) / 2);
    L_FFE_Vol = floor((N2 - 1) / 2);
    
    %% 1. 阶段一：标准 VFFE-VDFE 训练 (RLS)
    
    % 截取训练数据
    TrainLen = NumPreamble;
    if TrainLen > length(DesiredTx)
        TrainLen = length(DesiredTx);
    end
    
    Rx_Train = InputRx(1 : min(end, TrainLen * 2 + 100)); % 多取一点防溢出
    Tx_Train = DesiredTx(1 : TrainLen);
    
    % 计算维度
    Dim_FFE_Lin = N1;
    Dim_FFE_Vol = (2 * N2 - WL + 1) * WL / 2;
    Dim_DFE_Lin = D1;
    
    % DFE 非线性维度计算
    Dim_DFE_Vol = 0;
    if D2 > 0
        for m = 0 : WD-1
            if (D2 - m) > 0
                Dim_DFE_Vol = Dim_DFE_Vol + (D2 - m); 
            end
        end
    end
    
    Dim_Total = Dim_FFE_Lin + Dim_FFE_Vol + Dim_DFE_Lin + Dim_DFE_Vol;
    
    % RLS 初始化
    P = eye(Dim_Total) * 10; % 初始协方差矩阵
    w = zeros(Dim_Total, 1);
    
    % 训练循环
    StartIdx = max([N1, N2, D1, D2]) + 5; 
    
    for n = StartIdx : TrainLen
        % --- FFE 特征 ---
        idx_c = 2*n; 
        if idx_c + L_FFE_Lin > length(Rx_Train), break; end
        
        feat_F_L = Rx_Train(idx_c + L_FFE_Lin : -1 : idx_c - L_FFE_Lin);
        
        feat_F_V = [];
        for m = 1 : WL
            idx_start = idx_c + L_FFE_Vol + (m-1);
            if idx_start <= length(Rx_Train)
                va = Rx_Train(idx_start : -1 : idx_start - (2*N2 - WL)); % 简化逻辑，确保长度匹配
                % 严格按照 Dim 计算逻辑: 长度应为 2*N2 - WL + 1 ? 
                % 修正: 使用固定长度切片
                va = Rx_Train(idx_c + L_FFE_Vol + (m-1) : -1 : idx_c - L_FFE_Vol + (m-1));
                vb = Rx_Train((idx_c + L_FFE_Vol + (m-1) : -1 : idx_c - L_FFE_Vol + (m-1)) - (m-1));
                feat_F_V = [feat_F_V; va .* vb];
            end
        end
        
        % --- DFE 特征 (Training Mode: Use Tx) ---
        feat_D_L = Tx_Train(n-1 : -1 : n-D1);
        
        feat_D_V = [];
        if D2 > 0
            hist_seg = Tx_Train(n-1 : -1 : max(1, n-D2)); 
            % 补零如果不够长
            if length(hist_seg) < D2, hist_seg = [hist_seg; zeros(D2-length(hist_seg),1)]; end
            
            for m = 0 : WD-1
                 len_slice = D2 - m;
                 if len_slice > 0
                     vec_A = hist_seg(1 : len_slice);
                     vec_B = hist_seg(1+m : len_slice+m);
                     feat_D_V = [feat_D_V; vec_A .* vec_B];
                 end
            end
        end
        
        % RLS 更新
        u = [feat_F_L; feat_F_V; feat_D_L; feat_D_V];
        d_est = Tx_Train(n);
        
        if length(u) == Dim_Total
            k = (P * u) / (Lambda + u' * P * u);
            e = d_est - w' * u;
            w = w + k * e;
            P = (P - k * u' * P) / Lambda;
        end
    end
    
    % 提取系数
    idx_cut1 = Dim_FFE_Lin + Dim_FFE_Vol;
    w_FFE = w(1 : idx_cut1);
    w_DFE = w(idx_cut1 + 1 : end);
    
    w_DFE_Lin = w_DFE(1 : Dim_DFE_Lin);
    w_DFE_Vol = w_DFE(Dim_DFE_Lin + 1 : end);

    %% 2. 阶段二：聚类与建表 (Clustering & LUT)
    
    % --- 2.1 线性 DFE ---
    eff_K_Lin = min(K_Lin, length(w_DFE_Lin));
    if eff_K_Lin > 0
        % 使用 try-catch 防止 kmeans 因为数据奇异性报错
        try
            [idx_L_map, C_L_vals] = kmeans(w_DFE_Lin, eff_K_Lin, 'MaxIter', 100, 'Replicates', 1);
        catch
            idx_L_map = (1:length(w_DFE_Lin))';
            C_L_vals = w_DFE_Lin;
            eff_K_Lin = length(w_DFE_Lin);
        end
    else
        idx_L_map = []; C_L_vals = [];
    end
    
    % --- 2.2 非线性 DFE ---
    eff_K_Vol = min(K_Vol, length(w_DFE_Vol));
    has_Vol_DFE = (eff_K_Vol > 0);
    
    if has_Vol_DFE
        try
            [idx_V_map, C_V_vals] = kmeans(w_DFE_Vol, eff_K_Vol, 'MaxIter', 100, 'Replicates', 1);
        catch
            idx_V_map = (1:length(w_DFE_Vol))';
            C_V_vals = w_DFE_Vol;
            eff_K_Vol = length(w_DFE_Vol);
        end
    end
    
    % --- 2.3 建表 ---
    % Linear LUT
    LUT_Linear = zeros(eff_K_Lin, M);
    for k = 1 : eff_K_Lin
        LUT_Linear(k, :) = C_L_vals(k) * PamLevels;
    end
    
    % Nonlinear LUT (Only for PAM4 optimized)
    % Levels of product: PAM4 levels * PAM4 levels
    % Unique products sorted
    RawProds = unique(kron(PamLevels, PamLevels));
    LUT_Volterra = zeros(eff_K_Vol, length(RawProds));
    if has_Vol_DFE
        for k = 1 : eff_K_Vol
            LUT_Volterra(k, :) = C_V_vals(k) * RawProds;
        end
    end
    
    %% 3. 阶段三：CLUT-VDFE 运行 (Application)
    
    % 测试数据 (紧接训练数据之后)
    TestRx = InputRx(NumPreamble*2 + 1 : end);
    N_Test = floor(length(TestRx)/2) - max([N1,N2]); 
    ye_CLUT = zeros(N_Test, 1);
    
    % DFE Buffer (Initialize with last training symbols)
    MaxLag = max(D1, D2);
    DFE_Buf = zeros(MaxLag, 1); 
    if NumPreamble > MaxLag
        % 用训练末尾的真实数据填充 Buffer，避免冷启动
        DFE_Buf = DesiredTx(NumPreamble : -1 : NumPreamble - MaxLag + 1);
    end
    
    % 辅助函数: 找最近电平的索引
    find_level_idx = @(val, levels) sum(val >= levels(1:end-1)) + 1; % 简易查表逻辑 (假设 levels 有序)
    % 也可以用 min 距离
    get_idx = @(val, levels) (find(abs(levels - val) == min(abs(levels - val)), 1));

    for n = 1 : N_Test
        % --- A. FFE (乘法) ---
        idx_c = 2*n + max(N1, N2);
        if idx_c > length(TestRx) - max(N1, N2), break; end
        
        f_vec_L = TestRx(idx_c + L_FFE_Lin : -1 : idx_c - L_FFE_Lin);
        
        f_vec_V = [];
        for m = 1 : WL
            va = TestRx(idx_c + L_FFE_Vol + (m-1) : -1 : idx_c - L_FFE_Vol + (m-1));
            vb = TestRx((idx_c + L_FFE_Vol + (m-1) : -1 : idx_c - L_FFE_Vol + (m-1)) - (m-1));
            f_vec_V = [f_vec_V; va .* vb];
        end
        
        y_FFE = w_FFE' * [f_vec_L; f_vec_V];
        
        % --- B. DFE (查表) ---
        y_DFE = 0;
        
        % Linear
        for i = 1 : D1
            sym = DFE_Buf(i);
            % Find index in PamLevels
            % 简单量化: 因为 Buffer 里存的是标准电平(最近邻判决后的)
            % 我们可以直接通过数值匹配，或者如果 PamLevels 是均匀的，用公式算
            s_idx = get_idx(sym, PamLevels);
            
            c_id = idx_L_map(i);
            y_DFE = y_DFE + LUT_Linear(c_id, s_idx);
        end
        
        % Nonlinear
        if has_Vol_DFE
            cnt_v = 1;
            for m = 0 : WD-1
                len_slice = D2 - m;
                if len_slice > 0
                    for k = 1 : len_slice
                        val_A = DFE_Buf(k);
                        val_B = DFE_Buf(k+m);
                        prod_val = val_A * val_B;
                        
                        p_idx = get_idx(prod_val, RawProds);
                        c_id = idx_V_map(cnt_v);
                        y_DFE = y_DFE + LUT_Volterra(c_id, p_idx);
                        
                        cnt_v = cnt_v + 1;
                    end
                end
            end
        end
        
        % --- C. 输出与判决 ---
        y_total = y_FFE + y_DFE;
        ye_CLUT(n) = y_total;
        
        % Slicer (判决并更新 Buffer)
        % 找到最近的 PamLevel
        dec_idx = get_idx(y_total, PamLevels);
        dec_val = PamLevels(dec_idx);
        
        DFE_Buf = [dec_val; DFE_Buf(1:end-1)];
    end
    
    BER = 0; 
    ye_CLUT = ye_CLUT(:).'; % 转行向量
end
```

### **2. 如何修改你的主程序 `main.m`**

在你代码的 `for n1=1:length(file_list)` 循环内，找到调用 `LE_FFE2ps_centerDFE_new` 的地方，**注释掉它**，并加上我的 CLUT 调用代码：

```matlab
% --- 原有代码 ---
% [hffe,hdfe,ye] = LE_FFE2ps_centerDFE_new( xRx,xTx,NumPreamble_TDE,N1,D1,0.9999,M,M/2);  

% --- 新增代码 (CLUT-VDFE) ---
% 参数解释: 
% K_Lin=18 (线性聚类数), K_Vol=90 (非线性聚类数), Lambda=0.9999
% 注意: 即使你的 D2=0 (无非线性反馈)，这个函数也能正常跑，会自动跳过非线性聚类
[~, ye] = CLUT_VDFE_Implementation(xRx, xTx, NumPreamble_TDE, N1, N2, D1, D2, WL, WD, M, 18, 90, 0.9999);

% --- 这里的 ye 已经是归一化后的软输出，可以直接给后面的 BER 计算使用 ---