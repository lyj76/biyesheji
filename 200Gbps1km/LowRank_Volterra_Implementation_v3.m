function [ye_out, params_out, idxTx] = LowRank_Volterra_Implementation_v3(xRx, xTx, xsym, NumPreamble_TDE, Config)
    %% 0. 参数配置
    if nargin < 5
        Config = struct();
    end
    
    Rank = get_field(Config, 'Rank', 4);
    Volterra_Tap = get_field(Config, 'Tap', 15);
    LearningRate = get_field(Config, 'LR', [1e-3, 1e-4]); 
    MaxEpochs = get_field(Config, 'Epochs', 30);
    UseFeedback = get_field(Config, 'Feedback', true);
    UseWeightedLoss = get_field(Config, 'WeightedLoss', true);
    
    lr_p = LearningRate(1);
    lr_a = LearningRate(2);
    
    %% 1. Backbone: FFE Baseline (Strictly matching v2 logic)
    disp('    [Volterra-v3] Step 0: Linear FFE Baseline (Revert to v2 logic)...');
    
    % --- CRITICAL FIX: DO NOT NORMALIZE xRx/xTx HERE ---
    % Let FFE_2pscenter handle raw signals, just like in v2.
    
    N1 = 111; 
    Lambda = 0.9999;
    
    % FFE call matching v2
    [~, ye_lin_raw] = FFE_2pscenter(xRx, xTx, NumPreamble_TDE, N1, Lambda);
    
    M = 4;
    [off, d0] = align_offset_delay_by_ser(ye_lin_raw, xsym, NumPreamble_TDE, M, -60:60);
    
    if length(ye_lin_raw) > 1.5 * length(xsym)
        ye_lin_aligned = ye_lin_raw(off:2:end); 
    else
        ye_lin_aligned = ye_lin_raw;
    end
    
    N_Sym = length(ye_lin_aligned);
    idxTx = (1:N_Sym)' + d0;
    
    valid_mask = (idxTx >= 1) & (idxTx <= length(xTx));
    ye_lin = ye_lin_aligned(valid_mask);
    idxTx_valid = idxTx(valid_mask);
    xTx_valid = xTx(idxTx_valid);
    
    % 标准化 (Output-side only)
    scale_factor = std(xTx_valid);
    ye_lin_norm = ye_lin / scale_factor;
    ye_lin_norm = ye_lin_norm(:);
    xTx_norm = xTx_valid / scale_factor;
    xTx_norm = xTx_norm(:);
    
    %% 2. 准备 Volterra 输入
    pad_len = floor(Volterra_Tap/2);
    ye_padded = [zeros(pad_len, 1); ye_lin_norm; zeros(pad_len, 1)];
    
    N_Data = length(ye_lin_norm);
    U_Base = zeros(N_Data, Volterra_Tap);
    
    for i = 1:Volterra_Tap
        U_Base(:, i) = ye_padded(i : i + N_Data - 1);
    end
    
    %% 3. 初始化参数
    if UseFeedback
        Input_Dim = Volterra_Tap + 2; 
    else
        Input_Dim = Volterra_Tap;
    end
    
    P = randn(Input_Dim, Rank) * 0.05; 
    Alpha = zeros(Rank, 1); 
    
    %% 4. 训练
    disp(['    [Volterra-v3] Training | Rank=', num2str(Rank), ' | Tap=', num2str(Volterra_Tap)]);
    
    train_mask = idxTx_valid <= NumPreamble_TDE;
    
    U_Train = U_Base(train_mask, :);
    Target_Train = xTx_norm(train_mask);
    Y_Lin_Train = ye_lin_norm(train_mask);
    
    if UseFeedback
        T_pad = [0; 0; Target_Train];
        Fb1 = T_pad(2:end-1);
        Fb2 = T_pad(1:end-2);
        U_Train = [U_Train, Fb1, Fb2];
    end
    
    N_Train = length(Target_Train);
    BatchSize = 128;
    NumBatches = floor(N_Train / BatchSize);
    
    % Sample Weights Logic
    SampleWeights = ones(N_Train, 1);
    if UseWeightedLoss
        Thrs = [-0.9, 0, 0.9]';
        dists = abs(Y_Lin_Train - Thrs'); 
        min_dist = min(dists, [], 2);
        SampleWeights = 1 + 2.0 * exp(-min_dist / 0.2);
        SampleWeights = SampleWeights / mean(SampleWeights);
    end
    
    loss_history = [];
    
    for epoch = 1:MaxEpochs
        idx_shuffle = randperm(N_Train);
        epoch_loss = 0;
        
        for b = 1:NumBatches
            ids = idx_shuffle((b-1)*BatchSize+1 : b*BatchSize);
            
            u_batch = U_Train(ids, :);
            y_lin_batch = Y_Lin_Train(ids);
            tgt_batch = Target_Train(ids);
            w_batch = SampleWeights(ids);
            
            Z = u_batch * P;       
            Z_sq = Z.^2;           
            y_res = Z_sq * Alpha;  
            y_total = y_lin_batch + y_res;
            
            err = y_total - tgt_batch;
            grad_err = w_batch .* tanh(err); 
            
            epoch_loss = epoch_loss + sum(w_batch .* log(cosh(err)));
            
            grad_Alpha = Z_sq' * grad_err; 
            grad_P = zeros(size(P));
            for r = 1:Rank
                scale = grad_err .* (2 * Alpha(r) * Z(:, r));
                grad_P(:, r) = u_batch' * scale; 
            end
            
            grad_Alpha = grad_Alpha / BatchSize;
            grad_P = grad_P / BatchSize;
            
            Alpha = Alpha - lr_a * grad_Alpha;
            P = P - lr_p * grad_P;
            P = P * (1 - 1e-5); 
        end
        loss_history(end+1) = epoch_loss / N_Train;
    end
    disp(['    [Volterra-v3] Final Loss: ', num2str(loss_history(end))]);
    
    %% 5. 全局推断
    U_Test_Base = U_Base; 
    Y_Lin_Total = ye_lin_norm;
    N_Total = length(Y_Lin_Total);
    y_res_total = zeros(N_Total, 1);
    
    if ~UseFeedback
        Z_Full = U_Test_Base * P;
        y_res_total = (Z_Full.^2) * Alpha;
    else
        P_input = P(1:Volterra_Tap, :);
        P_fb    = P(Volterra_Tap+1:end, :);
        
        Z_input = U_Test_Base * P_input; 
        
        fb_regs = zeros(2, 1);
        thr = [-0.9, 0, 0.9];
        levels_norm = [-1.34, -0.447, 0.447, 1.34]; 
        
        for i = 1:N_Total
            z_fb = fb_regs' * P_fb;
            z_curr = Z_input(i, :) + z_fb;
            
            res_val = (z_curr.^2) * Alpha;
            y_res_total(i) = res_val;
            
            y_curr = Y_Lin_Total(i) + res_val;
            
            dec = slice_scalar(y_curr, levels_norm, thr);
            fb_regs(2) = fb_regs(1);
            fb_regs(1) = dec;
        end
    end
    
    ye_norm_out = Y_Lin_Total + y_res_total;
    ye_out = ye_norm_out * scale_factor; 
    
    params_out.P = P;
    params_out.Alpha = Alpha;
end

function val = get_field(struct, field, default)
    if isfield(struct, field)
        val = struct.(field);
    else
        val = default;
    end
end

function dec = slice_scalar(y, levels, thr)
    if y < thr(1), dec = levels(1);
    elseif y < thr(2), dec = levels(2);
    elseif y < thr(3), dec = levels(3);
    else, dec = levels(4);
    end
end