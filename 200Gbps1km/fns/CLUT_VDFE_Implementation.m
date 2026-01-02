function [BER, ye_aligned] = CLUT_VDFE_Implementation(Rx_in, Tx_in, NumPreamble, N1, N2, D1, D2, WL, WD, M, K_Lin, K_Vol, Lambda)
% CLUT_VDFE_Implementation: 修复版 - 保证全序列对齐
%
% 逻辑流:
% 1. 全序列处理 (Pad Rx).
% 2. n <= NumPreamble: RLS 训练模式 (Training).
% 3. n == NumPreamble: 触发聚类 & 建表 (Build LUT).
% 4. n >  NumPreamble: 查表模式 (CLUT Application).
% 5. 返回全长 ye，确保与主程序对齐.

    %% 0. 初始化
    InputRx = Rx_in(:);
    DesiredTx = Tx_in(:);
    
    % 归一化
    InputRx = InputRx ./ mean(abs(InputRx(:)));
    DesiredTx = DesiredTx ./ mean(abs(DesiredTx(:)));
    
    % 备份全长
    TotalSyms = length(DesiredTx);
    ye_aligned = zeros(TotalSyms, 1);
    
    % 辅助参数
    L_FFE_Lin = floor((N1 - 1) / 2);
    L_FFE_Vol = floor((N2 - 1) / 2);
    
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
    
    % RLS 变量
    P = eye(Dim_Total) * 10; 
    w = zeros(Dim_Total, 1);
    
    % LUT 变量 (初始化为空)
    LUT_Linear = [];
    LUT_Volterra = [];
    idx_L_map = [];
    idx_V_map = [];
    has_Vol_DFE = (D2 > 0);
    
    % PAM Levels (用于查表量化)
    if M == 4
        BaseLevels = [-3, -1, 1, 3];
        PamLevels = BaseLevels / mean(abs(BaseLevels)); 
    else
        BaseLevels = -(M-1):2:(M-1);
        PamLevels = BaseLevels / mean(abs(BaseLevels));
    end
    RawProds = unique(kron(PamLevels, PamLevels)); % 非线性积的电平表
    
    % 辅助函数: 找最近电平索引
    get_idx = @(val, levels) (find(abs(levels - val) == min(abs(levels - val)), 1));

    % 补零处理 (Align like DP_VFFE)
    PaddingLen = 1000;
    InputRx_Pad = [zeros(PaddingLen, 1); InputRx; zeros(L_FFE_Lin, 1)];
    DesiredTx_Pad = [zeros(PaddingLen/2, 1); DesiredTx];
    
    % DFE Buffer
    MaxLag = max(D1, D2) + 5;
    DFE_Buf = zeros(MaxLag, 1);

    % 主循环起始点
    StartIndex = PaddingLen/2 + 1; % 对应 DesiredTx(1)
    EndIndex = StartIndex + TotalSyms - 1;
    
    % 调试计数
    % disp('Starting CLUT-VDFE Loop...');
    
    %% 主循环 (Hybrid RLS -> CLUT)
    
    % loop_n 对应 DesiredTx 的索引 (1 .. TotalSyms)
    % pad_n  对应 Pad 后的索引
    
    for pad_n = StartIndex : EndIndex
        
        loop_n = pad_n - StartIndex + 1; % 当前处理的第几个符号
        
        % --- A. 构建 FFE 特征 (Common) ---
        idx_c = 2*pad_n; % 2 samples/symbol
        if idx_c + L_FFE_Lin > length(InputRx_Pad), break; end
        
        % 1. Linear FFE
        feat_F_L = InputRx_Pad(idx_c + L_FFE_Lin : -1 : idx_c - L_FFE_Lin);
        
        % 2. Volterra FFE
        feat_F_V = [];
        for m = 1 : WL
            idx_v_start = idx_c + L_FFE_Vol + (m-1);
            vec_a = InputRx_Pad(idx_v_start : -1 : idx_v_start - (2*N2 - WL)); 
            % 修正逻辑: 必须严格匹配 Dim 计算
            % 使用之前验证过的逻辑:
            idx_base = idx_c + L_FFE_Vol + (m-1);
            va = InputRx_Pad(idx_base : -1 : idx_base - (2*N2 - WL)); % 长度不对齐风险?
            % 让我们用绝对索引更安全:
            idx_s = idx_c + L_FFE_Vol + (m-1);
            idx_e = idx_c - L_FFE_Vol + (m-1);
            va = InputRx_Pad(idx_s : -1 : idx_e);
            vb = InputRx_Pad((idx_s : -1 : idx_e) - (m-1));
            feat_F_V = [feat_F_V; va .* vb];
        end
        
        % --- B. 模式分支 ---
        
        if loop_n <= NumPreamble
            
            % =========== RLS 训练模式 ===========
            
            % DFE 特征 (用 Tx)
            feat_D_L = DesiredTx_Pad(pad_n-1 : -1 : pad_n-D1);
            
            feat_D_V = [];
            if D2 > 0
                hist_seg = DesiredTx_Pad(pad_n-1 : -1 : pad_n-D2);
                % Pad if needed
                 if length(hist_seg) < D2, hist_seg = [hist_seg; zeros(D2-length(hist_seg),1)]; end

                for m = 0 : WD-1
                    if (D2-m) > 0
                        vec_a = hist_seg(1 : D2-m);
                        vec_b = hist_seg(1+m : D2);
                        feat_D_V = [feat_D_V; vec_a .* vec_b];
                    end
                end
            end
            
            % RLS 更新
            u = [feat_F_L; feat_F_V; feat_D_L; feat_D_V];
            d_target = DesiredTx_Pad(pad_n);
            
            y_out = w' * u;
            e = d_target - y_out;
            
            k = (P * u) / (Lambda + u' * P * u);
            w = w + k * e;
            P = (P - k * u' * P) / Lambda;
            
            ye_aligned(loop_n) = y_out;
            
            % 同步 Buffer (为切换做准备)
            if loop_n > 1
                DFE_Buf = [DesiredTx_Pad(pad_n-1); DFE_Buf(1:end-1)];
            end
            
            % =========== 触发点：聚类与建表 ===========
            if loop_n == NumPreamble
                % disp('Training done. Switching to CLUT mode...');
                
                % 提取系数
                idx_cut = Dim_FFE_Lin + Dim_FFE_Vol;
                w_FFE = w(1 : idx_cut);
                w_DFE = w(idx_cut+1 : end);
                w_DFE_Lin = w_DFE(1 : Dim_DFE_Lin);
                w_DFE_Vol = w_DFE(Dim_DFE_Lin+1 : end);
                
                % 1. Linear Clustering
                eff_K_Lin = min(K_Lin, length(w_DFE_Lin));
                try
                    [idx_L_map, C_L_vals] = kmeans(w_DFE_Lin, eff_K_Lin, 'Replicates', 1, 'MaxIter', 50);
                catch
                    idx_L_map = (1:length(w_DFE_Lin))'; C_L_vals = w_DFE_Lin; eff_K_Lin=length(w_DFE_Lin);
                end
                
                % 2. Volterra Clustering
                if has_Vol_DFE
                    eff_K_Vol = min(K_Vol, length(w_DFE_Vol));
                    try
                        [idx_V_map, C_V_vals] = kmeans(w_DFE_Vol, eff_K_Vol, 'Replicates', 1, 'MaxIter', 50);
                    catch
                        idx_V_map = (1:length(w_DFE_Vol))'; C_V_vals = w_DFE_Vol; eff_K_Vol=length(w_DFE_Vol);
                    end
                end
                
                % 3. Build LUTs
                LUT_Linear = zeros(eff_K_Lin, M);
                for k=1:eff_K_Lin, LUT_Linear(k,:) = C_L_vals(k) * PamLevels; end
                
                if has_Vol_DFE
                    LUT_Volterra = zeros(eff_K_Vol, length(RawProds));
                    for k=1:eff_K_Vol, LUT_Volterra(k,:) = C_V_vals(k) * RawProds; end
                end
            end
            
        else
            % =========== CLUT 查表模式 ===========
            
            % A. FFE (依然用乘法)
            % FFE 向量 u_FFE = [feat_F_L; feat_F_V]
            u_FFE = [feat_F_L; feat_F_V];
            y_FFE = w_FFE' * u_FFE;
            
            % B. DFE (用查表)
            y_DFE = 0;
            
            % B1. Linear LUT
            % Buffer(1) is x(n-1), corresponds to w_DFE_Lin(1)
            for i = 1 : D1
                sym = DFE_Buf(i);
                s_idx = get_idx(sym, PamLevels); % Quantize/Find Index
                c_id = idx_L_map(i);             % Find Cluster ID
                y_DFE = y_DFE + LUT_Linear(c_id, s_idx);
            end
            
            % B2. Nonlinear LUT
            if has_Vol_DFE
                cnt_v = 1;
                for m = 0 : WD-1
                    if (D2 - m) > 0
                        for k = 1 : (D2-m)
                            val_A = DFE_Buf(k);
                            val_B = DFE_Buf(k+m);
                            prod_val = val_A * val_B;
                            
                            p_idx = get_idx(prod_val, RawProds); % Find Product Index
                            c_id = idx_V_map(cnt_v);
                            y_DFE = y_DFE + LUT_Volterra(c_id, p_idx);
                            
                            cnt_v = cnt_v + 1;
                        end
                    end
                end
            end
            
            % Output
            y_out = y_FFE + y_DFE;
            ye_aligned(loop_n) = y_out;
            
            % Slicer & Update Buffer
            dec_idx = get_idx(y_out, PamLevels);
            dec_val = PamLevels(dec_idx);
            
            DFE_Buf = [dec_val; DFE_Buf(1:end-1)];
            
        end
    end
    
    BER = 0; % Placeholder
    ye_aligned = ye_aligned(:).';

end
