function [BER, ye_CLUT] = CLUT_VDFE_Implementation(Rx_in, Tx_in, NumPreamble, N1, N2, D1, D2, WL, WD, M, K_Lin, K_Vol, Lambda)
% CLUT_VDFE_Implementation: 修复版 - 统一索引与稳健电平学习
%
% 修复点:
% 1. 全局索引: 不再物理切分 InputRx，彻底解决 application 阶段开头的数据越界/对齐问题。
% 2. 真实电平学习: PamLevels 不再由公式生成，而是从训练数据中统计得出，防止归一化比例失调。
% 3. 调试信息: 输出 RLS 训练后的 MSE，判断训练是否收敛。

    %% 0. 数据预处理
    InputRx = Rx_in(:);
    DesiredTx = Tx_in(:);
    
    % --- 归一化 (关键: 保持 Rx 和 Tx 能量一致) ---
    scale_Rx = mean(abs(InputRx)); 
    if scale_Rx == 0, scale_Rx = 1; end
    InputRx = InputRx ./ scale_Rx;
    
    scale_Tx = mean(abs(DesiredTx));
    if scale_Tx == 0, scale_Tx = 1; end
    DesiredTx = DesiredTx ./ scale_Tx;
    
    % --- 学习真实的 PAM 电平 (用于 LUT) ---
    % 从训练序列的前 1000 个符号中提取聚类中心，确保查表电平与 RLS 目标完全一致
    % 这样即使 scale 稍有偏差，LUT 也能自动对齐
    if NumPreamble > 500
        ref_seq = DesiredTx(100:500); % 避开开头可能的暂态
    else
        ref_seq = DesiredTx;
    end
    [~, LearnedLevels] = kmeans(ref_seq, M, 'MaxIter', 10, 'Replicates', 3);
    PamLevels = sort(LearnedLevels).'; % 确保是从小到大排列 [-1.5, -0.5, ...]
    
    % 偏移量计算
    L_FFE_Lin = floor((N1 - 1) / 2);
    L_FFE_Vol = floor((N2 - 1) / 2);
    
    %% 1. 阶段一：RLS 训练 (Training Phase)
    
    % 维度计算
    Dim_FFE_Lin = N1;
    Dim_FFE_Vol = (2 * N2 - WL + 1) * WL / 2;
    Dim_DFE_Lin = D1;
    
    Dim_DFE_Vol = 0;
    if D2 > 0
        for m = 0 : WD-1
            if (D2 - m) > 0, Dim_DFE_Vol = Dim_DFE_Vol + (D2 - m); end
        end
    end
    
    Dim_Total = Dim_FFE_Lin + Dim_FFE_Vol + Dim_DFE_Lin + Dim_DFE_Vol;
    
    % RLS 初始化
    P = eye(Dim_Total) * 1; % 稍微减小初始方差，增加稳定性
    w = zeros(Dim_Total, 1);
    
    % 训练范围: 避开开头的 filter delay，直到 NumPreamble
    StartIdx = max([N1, N2, D1, D2]) + 10; 
    EndIdx_Train = NumPreamble;
    
    % 误差记录
    mse_history = zeros(EndIdx_Train - StartIdx + 1, 1);
    cnt = 1;

    for n = StartIdx : EndIdx_Train
        % 1.1 FFE 特征 (Rx)
        idx_c = 2*n; % 2 samples/symbol 假设对齐在偶数索引
        
        % 越界保护
        if idx_c + L_FFE_Lin > length(InputRx), break; end
        
        feat_F_L = InputRx(idx_c + L_FFE_Lin : -1 : idx_c - L_FFE_Lin);
        
        feat_F_V = [];
        for m = 1 : WL
            % 使用严格对齐的非线性项
            idx_base = idx_c + L_FFE_Vol + (m-1);
            va = InputRx(idx_base : -1 : idx_base - (2*N2 - WL)); % 长度修正
            % 上一行长度计算比较晦涩，改为显式循环构造或使用确定长度切片
            % 为了绝对稳健，这里改回最直观的写法:
            idx_start = idx_c + L_FFE_Vol + (m-1);
            idx_end = idx_c - L_FFE_Vol + (m-1);
            vec_raw = InputRx(idx_start : -1 : idx_end);
            
            vec_A = vec_raw;
            % vec_B 是 vec_A 延迟 m-1 个单位? 
            % 原代码逻辑: va .* vb. vb 是 va 延迟 m-1.
            % vec_B = InputRx((idx_start : -1 : idx_end) - (m-1));
            % 这等价于 vec_raw 本身和延迟后的自己相乘
            
            % 修正: 确保 vec_B 不越界
            vec_B = InputRx((idx_start - (m-1)) : -1 : (idx_end - (m-1)));
            
            feat_F_V = [feat_F_V; vec_A .* vec_B];
        end
        
        % 1.2 DFE 特征 (Tx - Ideal)
        feat_D_L = DesiredTx(n-1 : -1 : n-D1);
        
        feat_D_V = [];
        if D2 > 0
            % 必须使用一段连续的历史数据来构造交叉项，顺序极其重要
            hist_seg = DesiredTx(n-1 : -1 : max(1, n-D2));
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
        
        % 1.3 RLS Update
        u = [feat_F_L; feat_F_V; feat_D_L; feat_D_V];
        d_desired = DesiredTx(n);
        
        if length(u) == Dim_Total
            output_raw = w' * u;
            e = d_desired - output_raw;
            mse_history(cnt) = abs(e)^2; cnt = cnt + 1;
            
            k = (P * u) / (Lambda + u' * P * u);
            w = w + k * e;
            P = (P - k * u' * P) / Lambda;
        end
    end
    
    MSE_Final = mean(mse_history(max(1, end-500):end));
    disp(['  > CLUT-VDFE RLS Training MSE: ', num2str(MSE_Final)]);
    if MSE_Final > 0.5
        warning('RLS Training failed to converge (MSE > 0.5). Output may be garbage.');
    end
    
    % 提取系数
    idx_cut1 = Dim_FFE_Lin + Dim_FFE_Vol;
    w_FFE = w(1 : idx_cut1);
    w_DFE = w(idx_cut1 + 1 : end);
    w_DFE_Lin = w_DFE(1 : Dim_DFE_Lin);
    w_DFE_Vol = w_DFE(Dim_DFE_Lin + 1 : end);

    %% 2. 阶段二：聚类与建表 (Clustering & LUT)
    
    % --- 2.1 线性 DFE ---
    % 如果 K_Lin >= D1，不进行聚类，直接一对一映射（退化为普通 LUT）
    eff_K_Lin = min(K_Lin, length(w_DFE_Lin));
    
    if eff_K_Lin < length(w_DFE_Lin)
        try
            [idx_L_map, C_L_vals] = kmeans(w_DFE_Lin, eff_K_Lin, 'MaxIter', 100, 'Replicates', 1);
        catch
            warning('K-means failed for Linear DFE. Fallback to full coefficients.');
            idx_L_map = (1:length(w_DFE_Lin))'; C_L_vals = w_DFE_Lin; eff_K_Lin = length(w_DFE_Lin);
        end
    else
        idx_L_map = (1:length(w_DFE_Lin))'; C_L_vals = w_DFE_Lin;
    end
    
    % --- 2.2 非线性 DFE ---
    eff_K_Vol = min(K_Vol, length(w_DFE_Vol));
    has_Vol_DFE = (eff_K_Vol > 0);
    
    if has_Vol_DFE && eff_K_Vol < length(w_DFE_Vol)
        try
            [idx_V_map, C_V_vals] = kmeans(w_DFE_Vol, eff_K_Vol, 'MaxIter', 100, 'Replicates', 1);
        catch
            idx_V_map = (1:length(w_DFE_Vol))'; C_V_vals = w_DFE_Vol; eff_K_Vol = length(w_DFE_Vol);
        end
    elseif has_Vol_DFE
        idx_V_map = (1:length(w_DFE_Vol))'; C_V_vals = w_DFE_Vol;
    end
    
    % --- 2.3 建表 ---
    % 线性表: Rows=Cluster, Cols=Levels
    LUT_Linear = zeros(eff_K_Lin, M);
    for k = 1 : eff_K_Lin
        LUT_Linear(k, :) = C_L_vals(k) * PamLevels;
    end
    
    % 非线性表: 构建积的可能值
    % PamLevels 包含从训练数据学到的电平，例如 [-1.4, -0.4, 0.4, 1.4]
    % 我们需要所有可能的两两乘积
    RawProds = unique(kron(PamLevels, PamLevels));
    LUT_Volterra = zeros(eff_K_Vol, length(RawProds));
    if has_Vol_DFE
        for k = 1 : eff_K_Vol
            LUT_Volterra(k, :) = C_V_vals(k) * RawProds;
        end
    end
    
    %% 3. 阶段三：CLUT-VDFE 运行 (Application Phase)
    
    % 关键修复: 不切分 Rx，而是从 NumPreamble + 1 开始继续跑
    % 这样历史数据 (FFE window) 自然存在于 InputRx 中
    StartIdx_App = NumPreamble + 1;
    EndIdx_App = length(DesiredTx); 
    
    N_Test = EndIdx_App - StartIdx_App + 1;
    ye_CLUT = zeros(N_Test, 1);
    
    % 初始化 Buffer (用最后一段训练数据热启动)
    MaxLag = max(D1, D2);
    if StartIdx_App > MaxLag
        DFE_Buf = DesiredTx(StartIdx_App-1 : -1 : StartIdx_App-MaxLag);
    else
        DFE_Buf = zeros(MaxLag, 1);
    end
    
    % 快速查找辅助函数 (Nearest Neighbor)
    % 给定一个值 val，在 vec 中找到最近值的索引
    get_idx = @(val, vec) (find(abs(vec - val) == min(abs(vec - val)), 1));

    disp('  > Running CLUT-VDFE Application...');
    
    for i = 1 : N_Test
        n = StartIdx_App + (i-1); % 当前处理的绝对符号索引
        
        % --- A. FFE (Standard Calculation) ---
        idx_c = 2*n; 
        
        if idx_c + L_FFE_Lin > length(InputRx), break; end
        
        feat_F_L = InputRx(idx_c + L_FFE_Lin : -1 : idx_c - L_FFE_Lin);
        
        feat_F_V = [];
        for m = 1 : WL
            idx_start = idx_c + L_FFE_Vol + (m-1);
            idx_end = idx_c - L_FFE_Vol + (m-1);
            
            vec_A = InputRx(idx_start : -1 : idx_end);
            vec_B = InputRx((idx_start - (m-1)) : -1 : (idx_end - (m-1)));
            feat_F_V = [feat_F_V; vec_A .* vec_B];
        end
        
        y_FFE = w_FFE' * [feat_F_L; feat_F_V];
        
        % --- B. DFE (CLUT - No Multiplication) ---
        y_DFE = 0;
        
        % 线性部分查表
        for k = 1 : D1
            sym_val = DFE_Buf(k);
            % 找到该值对应 PamLevels 的第几个 (1..M)
            s_idx = get_idx(sym_val, PamLevels);
            
            cluster_id = idx_L_map(k);
            y_DFE = y_DFE + LUT_Linear(cluster_id, s_idx);
        end
        
        % 非线性部分查表
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
                        cluster_id = idx_V_map(cnt_v);
                        y_DFE = y_DFE + LUT_Volterra(cluster_id, p_idx);
                        
                        cnt_v = cnt_v + 1;
                    end
                end
            end
        end
        
        % --- C. 总输出与判决 ---
        y_total = y_FFE + y_DFE;
        ye_CLUT(i) = y_total;
        
        % Slicer: 判决为最近的 PamLevel
        dec_idx = get_idx(y_total, PamLevels);
        dec_val = PamLevels(dec_idx);
        
        % 更新 Buffer
        DFE_Buf = [dec_val; DFE_Buf(1:end-1)];
    end
    
    BER = 0; % 外部计算
    ye_CLUT = ye_CLUT(:).'; % 保持行向量
    
    % 反归一化 (可选，为了匹配外部 BER 计算量级)
    % ye_CLUT = ye_CLUT * scale_Tx; 
    % 但通常外部 BER 计算会自动处理归一化，这里返回标准化的 ye 更安全
end