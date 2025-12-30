function [ye_out, params_out, idxTx] = LowRank_Volterra_Implementation_v2(xRx, xTx, xsym, NumPreamble_TDE, Rank, LearningRate, MaxEpochs, UseFeedback)
    %% 0. 参数配置
    if nargin < 5, Rank = 2; end
    if nargin < 6, LearningRate = [1e-3, 1e-4]; end % [lr_p, lr_alpha] (稍微调大一点)
    if nargin < 7, MaxEpochs = 30; end
    if nargin < 8, UseFeedback = false; end
    
    lr_p = LearningRate(1);
    lr_a = LearningRate(2);
    
    %% 1. Backbone: 使用成熟的 FFE 获取基准 (稳健之源)
    disp('    [Volterra] Step 0: Running Linear FFE Baseline...');
    
    % FFE 参数 (和你主程序保持一致)
    N1 = 111; 
    Lambda = 0.9999;
    
    % 调用 FFE_2pscenter (它返回的 ye 已经是经过滤波的，但可能还需要对齐)
    [~, ye_lin_raw] = FFE_2pscenter(xRx, xTx, NumPreamble_TDE, N1, Lambda);
    
    % 统一对齐 (至关重要)
    M = 4;
    [off, d0] = align_offset_delay_by_ser(ye_lin_raw, xsym, NumPreamble_TDE, M, -60:60);
    
    % 降采样/截断 (处理 2pscenter 输出可能的过采样或 shift)
    % FFE_2pscenter 输出通常已经是 T 间隔 (1 sps)
    % 但 align_offset_delay_by_ser 可能会建议 offset
    if length(ye_lin_raw) > 1.5 * length(xsym)
        % 如果 FFE 输出是 2sps，才需要降采样，但 FFE_2pscenter 应该已经是 1sps
        % 这里为了保险，按常规逻辑处理
        ye_lin_aligned = ye_lin_raw(off:2:end); 
    else
        ye_lin_aligned = ye_lin_raw;
    end
    
    % 构建 idxTx 映射
    N_Sym = length(ye_lin_aligned);
    idxTx = (1:N_Sym)' + d0;
    
    % 只保留有效部分
    valid_mask = (idxTx >= 1) & (idxTx <= length(xTx));
    ye_lin = ye_lin_aligned(valid_mask);
    idxTx_valid = idxTx(valid_mask);
    xTx_valid = xTx(idxTx_valid); % 对应的发送符号 (理想目标)
    
    % 标准化基准信号 (便于后续 NN 训练)
    % 我们希望残差是在一个归一化的尺度上学习
    scale_factor = std(xTx_valid);
    ye_lin_norm = ye_lin / scale_factor;
    ye_lin_norm = ye_lin_norm(:); % Force column vector
    xTx_norm = xTx_valid / scale_factor;
    
    % 计算 Backbone 误差 (这就是我们要去修补的 Residual)
    % E_target = xTx_norm - ye_lin_norm; 
    % (不需要显式存 E_target，我们在循环里算，因为 feedback 模式下 y_total 会变)
    
    %% 2. 准备 Volterra 输入数据
    % 我们需要从 xRx 中提取对应的输入向量 u
    % 注意：由于时序经过了 FFE 和对齐，直接从 xRx 取数比较麻烦。
    % 
    % --- 极其聪明的 Trick ---
    % 既然 FFE 已经是线性的，且我们要学的是“非线性修正”。
    % 我们可以直接用 **FFE 的输出 ye_lin** 作为 Volterra 的输入源！
    % 原因：ye_lin 已经包含了主要的信号特征，只是带有一些残余 ISI 和非线性失真。
    % 用 ye_lin 周围的几个点来预测当前点的非线性修正，效果通常足够好，且无需回溯原始 xRx。
    % 
    % 输入向量 u[n] = [ye_lin[n], ye_lin[n-1], ye_lin[n-2], ..., ye_lin[n-4]]
    
    Volterra_Tap = 5;
    
    % 构建 Hankel 矩阵 U
    % Padding 以保持长度一致
    pad_len = floor(Volterra_Tap/2);
    ye_padded = [zeros(pad_len, 1); ye_lin_norm; zeros(pad_len, 1)];
    
    N_Data = length(ye_lin_norm);
    U_Base = zeros(N_Data, Volterra_Tap);
    
    % 构造输入矩阵 (滑动窗口)
    % 中心对齐：ye_lin[n] 对应 U_Base(n, center)
    % 这里我们简单取 [n, n-1, n-2, n-3, n-4] 这种因果或非因果窗都可以
    % 既然 ye_lin 已经消除了大部分 ISI，我们取以当前点为中心的窗
    % [n-2, n-1, n, n+1, n+2]
    for i = 1:Volterra_Tap
        % shift
        U_Base(:, i) = ye_padded(i : i + N_Data - 1);
    end
    
    %% 3. 初始化参数
    if UseFeedback
        Input_Dim = Volterra_Tap + 2; 
    else
        Input_Dim = Volterra_Tap;
    end
    
    P = randn(Input_Dim, Rank) * 0.05; % 小随机初始化
    Alpha = zeros(Rank, 1); % 初始为 0 (线性保底)
    
    %% 4. 训练 (只用 Preamble)
    disp(['    [Volterra] Training (Log-Cosh Loss) | Rank=', num2str(Rank), ' | Feedback=', num2str(UseFeedback)]);
    
    % 划分训练集 (Preamble)
    % idxTx_valid 中，小于 NumPreamble_TDE 的是训练集
    train_mask = idxTx_valid <= NumPreamble_TDE;
    test_mask  = idxTx_valid > NumPreamble_TDE;
    
    if sum(train_mask) < 1000
        warning('Training samples too few! Check alignment.');
    end
    
    U_Train = U_Base(train_mask, :);
    Target_Train = xTx_norm(train_mask);
    Y_Lin_Train = ye_lin_norm(train_mask);
    
    % Feedback 用的 Teacher Signal (理想符号作为反馈)
    if UseFeedback
        % 构造反馈矩阵 (Ideal Feedback for Training)
        % Fb1 = Target[n-1], Fb2 = Target[n-2]
        T_pad = [0; 0; Target_Train];
        Fb1 = T_pad(2:end-1);
        Fb2 = T_pad(1:end-2);
        U_Train = [U_Train, Fb1, Fb2];
    end
    
    N_Train = length(Target_Train);
    BatchSize = 64;
    NumBatches = floor(N_Train / BatchSize);
    
    loss_history = [];
    
    for epoch = 1:MaxEpochs
        idx_shuffle = randperm(N_Train);
        epoch_loss = 0;
        
        for b = 1:NumBatches
            ids = idx_shuffle((b-1)*BatchSize+1 : b*BatchSize);
            
            u_batch = U_Train(ids, :);
            y_lin_batch = Y_Lin_Train(ids);
            tgt_batch = Target_Train(ids);
            
            % --- Forward ---
            Z = u_batch * P;       % [B x R]
            Z_sq = Z.^2;           % [B x R]
            y_res = Z_sq * Alpha;  % [B x 1]
            y_total = y_lin_batch + y_res;
            
            % --- Loss: Log-Cosh ---
            err = y_total - tgt_batch;
            
            % dLoss/derr = tanh(err)
            grad_err = tanh(err); % [B x 1] (Robust Gradient!)
            
            epoch_loss = epoch_loss + sum(log(cosh(err)));
            
            % --- Backward ---
            % 1. dL/dAlpha = grad_err * Z^2
            grad_Alpha = Z_sq' * grad_err; % [R x 1]
            
            % 2. dL/dP = grad_err * 2 * Alpha * Z * U
            grad_P = zeros(size(P));
            for r = 1:Rank
                % scale = grad_err * 2 * Alpha(r) * Z(:,r)
                scale = grad_err .* (2 * Alpha(r) * Z(:, r));
                grad_P(:, r) = u_batch' * scale; 
            end
            
            % Average
            grad_Alpha = grad_Alpha / BatchSize;
            grad_P = grad_P / BatchSize;
            
            % Update
            Alpha = Alpha - lr_a * grad_Alpha;
            P = P - lr_p * grad_P;
        end
        loss_history(end+1) = epoch_loss / N_Train;
    end
    disp(['    [Volterra] Final Loss: ', num2str(loss_history(end))]);
    
    %% 5. 全局推断
    % 对整个序列进行推断
    U_Test_Base = U_Base; % 包含训练和测试的所有数据
    Y_Lin_Total = ye_lin_norm;
    
    if ~UseFeedback
        Z_Full = U_Test_Base * P;
        y_res_total = (Z_Full.^2) * Alpha;
    else
        % 有反馈的推理 (需要逐个符号跑 DFE)
        N_Total = length(Y_Lin_Total);
        y_res_total = zeros(N_Total, 1);
        
        P_input = P(1:Volterra_Tap, :);
        P_fb    = P(Volterra_Tap+1:end, :);
        
        Z_input = U_Test_Base * P_input; % 预计算
        
        fb_regs = zeros(2, 1);
        
        % 估计电平用于硬判决反馈
        % 归一化后的 PAM4 电平大致是 -1.34, -0.45, 0.45, 1.34 (Z-score)
        % 简单起见，直接用 3-Level 聚类或固定阈值
        % 这里用固定阈值: -0.9, 0, 0.9 (近似值)
        thr = [-0.9, 0, 0.9];
        levels_norm = [-1.34, -0.447, 0.447, 1.34]; % 理论 Z-score 值
        
        for i = 1:N_Total
            z_fb = fb_regs' * P_fb;
            z_curr = Z_input(i, :) + z_fb;
            
            res_val = (z_curr.^2) * Alpha;
            y_res_total(i) = res_val;
            
            y_curr = Y_Lin_Total(i) + res_val;
            
            % 简单切片
            dec = slice_scalar(y_curr, levels_norm, thr);
            fb_regs(2) = fb_regs(1);
            fb_regs(1) = dec;
        end
    end
    
    % 合并与还原
    ye_norm_out = Y_Lin_Total + y_res_total;
    ye_out = ye_norm_out * scale_factor; % 还原幅度
    
    params_out.P = P;
    params_out.Alpha = Alpha;
    
    % idxTx 已经包含了 valid mask 的筛选
    % 但函数需要返回完整的 output 对应关系
    % idxTx 变量名其实是 valid_idx，直接返回即可
end

function dec = slice_scalar(y, levels, thr)
    if y < thr(1), dec = levels(1);
    elseif y < thr(2), dec = levels(2);
    elseif y < thr(3), dec = levels(3);
    else, dec = levels(4);
    end
end
