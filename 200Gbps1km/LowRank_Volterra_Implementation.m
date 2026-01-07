function [ye_out, params_out, idxTx] = LowRank_Volterra_Implementation(xRx, xTx, NumPreamble_TDE, FFE_Len, Rank, LearningRate, MaxEpochs, UseFeedback)
    %% 0. 参数配置
    if nargin < 4, FFE_Len = 111; end
    if nargin < 5, Rank = 2; end
    if nargin < 6, LearningRate = [1e-4, 1e-5]; end % [lr_p, lr_alpha]
    if nargin < 7, MaxEpochs = 20; end
    if nargin < 8, UseFeedback = false; end
    
    lr_p = LearningRate(1);
    lr_a = LearningRate(2);
    
    %% 1. 数据对齐与预处理
    % 这里我们用之前最稳健的方法：先用线性 FFE 扫一个最佳 delay
    % 但为了代码独立，我们这里重写一个简化的线性对齐
    
    Rx_Data = xRx(:);
    Tx_Data = xTx(:);
    
    % 标准化 (Z-score)
    Rx_mean = mean(Rx_Data); Rx_std = std(Rx_Data);
    Tx_mean = mean(Tx_Data); Tx_std = std(Tx_Data);
    
    Rx_Norm = (Rx_Data - Rx_mean) / Rx_std;
    Tx_Norm = (Tx_Data - Tx_mean) / Tx_std;
    
    % 对齐：简单的互相关粗对齐 + 线性探测精对齐
    % 粗对齐
    xc = xcorr(Rx_Norm(1:5000), Tx_Norm(1:5000));
    [~, lag] = max(abs(xc));
    delay_coarse = lag - 5000;
    
    % 这一步如果不对，后面都白搭。所以我们还是用一个标准的线性最小二乘来定 Delay
    disp('    [Volterra] Pre-alignment (Linear LS)...');
    best_mse = inf;
    best_d = 0;
    FFE_Half = floor(FFE_Len/2);
    
    % 搜索范围：粗对齐附近 +/- 20
    search_range = (delay_coarse - 20) : (delay_coarse + 20);
    
    % 仅用 Preamble 的一小部分来定 delay
    N_Probe = min(NumPreamble_TDE, 5000); 
    
    for d = search_range
        % 构造 X 矩阵
        [X_probe, Y_probe, ~] = build_data_matrix(Rx_Norm, Tx_Norm, FFE_Len, d, N_Probe);
        if size(X_probe, 2) < FFE_Len, continue; end
        
        % LS 解
        w_tmp = (X_probe' * X_probe + 1e-4*eye(FFE_Len)) \ (X_probe' * Y_probe);
        mse = mean((X_probe * w_tmp - Y_probe).^2);
        
        if mse < best_mse
            best_mse = mse;
            best_d = d;
            w_best_linear = w_tmp; % 保存这个线性权重作为 Backbone
        end
    end
    disp(['    [Volterra] Best Delay: ', num2str(best_d), ', Linear MSE: ', num2str(best_mse)]);
    
    %% 2. 准备训练数据
    % Backbone 固定为 w_best_linear
    % 我们只在残差上做 SGD
    
    [X_All, Y_All, idxTx] = build_data_matrix(Rx_Norm, Tx_Norm, FFE_Len, best_d, NumPreamble_TDE);
    
    % 计算 Backbone 输出
    Y_Lin = X_All * w_best_linear;
    
    % 目标残差 (Target Residual)
    E_Target = Y_All - Y_Lin;
    
    %% 3. 初始化 Volterra 参数
    % 输入向量 u 的长度
    % 如果无反馈：u = [x[n], x[n-1], x[n-2], x[n-3], x[n-4]] (5 taps)
    % 如果有反馈：u = [x[n]...x[n-4], a[n-1], a[n-2]] (7 taps)
    
    Volterra_Tap = 5;
    if UseFeedback
        Input_Dim = Volterra_Tap + 2; % 加 2 个反馈
    else
        Input_Dim = Volterra_Tap;
    end
    
    % 初始化 p (投影向量) - 高斯随机
    P = randn(Input_Dim, Rank) * 0.1; 
    
    % 初始化 alpha (组合系数) - 设为 0，从线性模型开始
    Alpha = zeros(Rank, 1); 
    
    % 提取 u 向量所需的索引偏移 (相对于 FFE 中心)
    % FFE 中心是 FFE_Half + 1
    % 我们取中心附近的 5 个点作为 Volterra 输入
    center_tap = FFE_Half + 1;
    v_idx_offset = (center_tap - floor(Volterra_Tap/2)) : (center_tap + floor(Volterra_Tap/2));
    
    %% 4. 手写 SGD 训练循环
    disp(['    [Volterra] Training Residual Model (Rank=', num2str(Rank), ', Feedback=', num2str(UseFeedback), ')...']);
    
    N_Train = length(Y_All);
    Loss_History = zeros(MaxEpochs, 1);
    
    % 预先准备好 Soft Feedback 的容器 (如果是 Feedback 模式)
    if UseFeedback
        % 既然是 Block 处理，为了简化 SGD，我们这里先用 "Teacher Forcing"
        % 即训练时使用真实的 label 作为反馈，推理时才用估计值
        % 这样可以让 SGD 极快并行化 (虽然这里是 for 循环)
        % 真实的反馈符号
        Feedback_Syms = Y_All; % 注意：这里用 Y_All (Target) 近似上一时刻的判决
    end

    % 转换 X_All 的特定列作为 Volterra 输入 U_Base
    % X_All 的每一行是 [x[n], x[n-1] ... ]
    % 我们取对应的列
    U_Base = X_All(:, v_idx_offset); % [N x 5]
    
    % 开始 Epoch
    for epoch = 1:MaxEpochs
        total_loss = 0;
        
        % SGD 逐样本更新 (或者 Mini-batch)
        % 为了速度，这里用 Mini-batch (size=32)
        BatchSize = 32;
        NumBatches = floor(N_Train / BatchSize);
        
        idx_shuffle = randperm(N_Train);
        
        for b = 1:NumBatches
            % 获取 Batch 数据
            ids = idx_shuffle((b-1)*BatchSize+1 : b*BatchSize);
            
            e_tgt_batch = E_Target(ids); % [B x 1]
            
            % 构造 U
            u_batch = U_Base(ids, :); % [B x 5]
            
            if UseFeedback
                % 添加反馈项
                % 真实的反馈: Y_All 是 target (理想符号)
                % 我们需要 delay 1 和 delay 2 的 target
                % 注意：这里处理边界有点麻烦，简单起见，如果 idx < 2 就补 0
                % 向量化处理:
                fb1 = zeros(BatchSize, 1);
                fb2 = zeros(BatchSize, 1);
                
                % 这里的 ids 是在 X_All 中的行号
                % X_All 的第 k 行对应 Y_All(k)
                % 所以上一时刻是 Y_All(k-1)
                
                valid_fb1 = ids > 1;
                fb1(valid_fb1) = Y_All(ids(valid_fb1)-1);
                
                valid_fb2 = ids > 2;
                fb2(valid_fb2) = Y_All(ids(valid_fb2)-2);
                
                u_batch = [u_batch, fb1, fb2]; % [B x 7]
            end
            
            % --- 前向传播 ---
            % Z = U * P; % [B x Rank]
            Z = u_batch * P;
            
            % Y_Res = sum(Alpha' .* Z.^2, 2)
            % 更加显式的写：
            Z_sq = Z.^2; % [B x Rank]
            y_res_batch = Z_sq * Alpha; % [B x 1]
            
            % --- 误差 ---
            % Loss = 0.5 * (y_res - e_tgt)^2
            % grad_L_y = (y_res - e_tgt)
            batch_err = y_res_batch - e_tgt_batch; % [B x 1]
            
            total_loss = total_loss + sum(batch_err.^2);
            
            % --- 反向传播 (梯度计算) ---
            % 1. 对 Alpha 的梯度
            % dL/dAlpha = err * Z^2
            grad_Alpha = Z_sq' * batch_err; % [Rank x 1]
            
            % 2. 对 P 的梯度
            % dL/dP_r = err * 2 * Alpha_r * Z_r * U
            % 这是一个 [Input_Dim x Rank] 的矩阵
            grad_P = zeros(size(P));
            for r = 1:Rank
                % elem-wise: err .* (2 * Alpha(r) * Z(:,r)) -> [B x 1]
                % 然后乘 U_batch
                scale = batch_err .* (2 * Alpha(r) * Z(:, r));
                grad_P(:, r) = u_batch' * scale;
            end
            
            % 平均梯度
            grad_Alpha = grad_Alpha / BatchSize;
            grad_P = grad_P / BatchSize;
            
            % --- 参数更新 ---
            Alpha = Alpha - lr_a * grad_Alpha;
            P = P - lr_p * grad_P;
            
            % 投影归一化 (可选，防止发散)
            % P = P ./ max(abs(P), 1e-2); 
        end
        
        Loss_History(epoch) = total_loss / N_Train;
        % disp(['    Epoch ', num2str(epoch), ' Loss: ', num2str(Loss_History(epoch))]);
    end
    disp(['    [Volterra] Final Loss: ', num2str(Loss_History(end))]);
    
    params_out.P = P;
    params_out.Alpha = Alpha;
    params_out.w_lin = w_best_linear;
    params_out.best_d = best_d;
    params_out.v_idx_offset = v_idx_offset;
    params_out.UseFeedback = UseFeedback;
    params_out.Volterra_Tap = Volterra_Tap;
    
    %% 5. 全局推断 (Inference)
    disp('    [Volterra] Full Inference...');
    
    % 构建全量测试数据
    [X_Full, ~, idxTx] = build_data_matrix(Rx_Norm, Tx_Norm, FFE_Len, best_d, []);
    
    % 1. Backbone Output
    y_lin_full = X_Full * w_best_linear;
    
    % 2. Residual Output
    N_Full = size(X_Full, 1);
    y_res_full = zeros(N_Full, 1);
    
    U_Full_Base = X_Full(:, v_idx_offset);
    
    if ~UseFeedback
        % 无反馈，直接矩阵运算 (极快)
        Z_Full = U_Full_Base * P;
        y_res_full = (Z_Full.^2) * Alpha;
    else
        % 有反馈，必须逐符号循环 (因为当前输出依赖上一时刻的判决)
        % 这就是 DFE 慢的原因，但我们模型小，应该还好
        
        % 预计算部分 (Input 部分的投影)
        % P 分成两部分：P_input [5 x R], P_fb [2 x R]
        P_input = P(1:Volterra_Tap, :);
        P_fb    = P(Volterra_Tap+1:end, :);
        
        Z_input = U_Full_Base * P_input; % [N x R] 预先算好
        
        % 循环
        % 初始化反馈寄存器 (soft symbols or hard decisions)
        fb_regs = zeros(2, 1); 
        
        % 为了加速，我们可以用 slice_levels 做硬判决反馈
        % 这里需要知道电平，简单假设 [-3 -1 1 3] (归一化后要调整)
        % 既然是 Z-score 归一化，Tx ~ N(0,1), -3 -> -1.34, 3 -> 1.34
        % 简单用 sign 判决? 不够，PAM4
        % 我们动态估计一下 levels
        % 其实可以用 y_lin_full 来辅助判决
        
        % 快速聚类找电平
        temp_y = y_lin_full(1:min(1000, end));
        [~, C] = kmeans(double(temp_y), 4, 'Replicates', 3);
        levels = sort(C);
        thr = (levels(1:3) + levels(2:4))/2;
        
        for i = 1:N_Full
            % 当前时刻的 Z
            % Z[i, r] = Z_input[i, r] + fb' * P_fb[:, r]
            
            % 反馈向量 fb_regs: [a[n-1]; a[n-2]]
            z_fb = fb_regs' * P_fb; % [1 x R]
            
            z_curr = Z_input(i, :) + z_fb;
            
            % 残差
            res_val = (z_curr.^2) * Alpha;
            y_res_full(i) = res_val;
            
            % 总输出
            y_curr = y_lin_full(i) + res_val;
            
            % 判决 (更新反馈)
            % 简单 slice
            dec = slice_scalar(y_curr, levels, thr);
            
            % 移位寄存器
            fb_regs(2) = fb_regs(1);
            fb_regs(1) = dec;
        end
    end
    
    % 合并
    ye_total = y_lin_full + y_res_full;
    
    % 反归一化
    ye_out = ye_total * Tx_std + Tx_mean;
    idxTx = idxTx(:);
end

function dec = slice_scalar(y, levels, thr)
    if y < thr(1), dec = levels(1);
    elseif y < thr(2), dec = levels(2);
    elseif y < thr(3), dec = levels(3);
    else, dec = levels(4);
    end
end

function [X, Y, idx] = build_data_matrix(Rx, Tx, Len, delay, max_samples)
    N = length(Tx);
    idx_tx = (1:N)';
    
    % Rx center index for Tx[n] is n + delay
    % Window: (n + delay) - floor(Len/2) ...
    half = floor(Len/2);
    center = idx_tx + delay;
    
    start_idx = center - half;
    end_idx = center + half; % if Len is odd
    if mod(Len, 2) == 0, end_idx = start_idx + Len - 1; end
    
    valid = start_idx >= 1 & end_idx <= length(Rx) & idx_tx >= 1 & idx_tx <= N;
    
    if ~isempty(max_samples)
        % limit samples
        valid_idx_list = find(valid);
        if length(valid_idx_list) > max_samples
            valid_idx_list = valid_idx_list(1:max_samples);
            % reset valid mask
            valid = false(size(valid));
            valid(valid_idx_list) = true;
        end
    end
    
    idx = idx_tx(valid);
    starts = start_idx(valid);
    
    num_valid = length(idx);
    X = zeros(num_valid, Len);
    
    % Hankel matrix construction (can be optimized but loop is fine for offline)
    for i = 1:num_valid
        X(i, :) = Rx(starts(i) : starts(i)+Len-1);
    end
    
    Y = Tx(idx);
end