function [h, d, ye] = DP_VFFE2pscenter_VDFE(xTx_in, xRx_in, NumPreamble_TDE, N1, N2, D1, D2, Lambda, WL, WD, M, scale)
% DP_VFFE2pscenter_VDFE: 全非线性判决反馈均衡器 (Volterra FFE + Volterra DFE)
%
% 这是一个结合了 Volterra 非线性前馈 (V-FFE) 和 Volterra 非线性反馈 (V-DFE) 的高级均衡器。
%
% 输入变量说明:
%   xTx_in:          接收到的信号 (Input Rx), 2 samples/symbol (过采样)
%   xRx_in:          期望的训练序列 (Desired Tx), 1 sample/symbol
%   NumPreamble_TDE: 训练序列长度
%   N1:              FFE 线性抽头长度 (Linear Memory Depth)
%   N2:              FFE 非线性项的覆盖范围 (Nonlinear Memory Depth)
%   D1:              DFE 线性抽头长度 (Feedback Linear Memory)
%   D2:              DFE 非线性项的覆盖范围 (Feedback Nonlinear Memory)。
%                    这是用来控制“我们回溯多远的历史判决来计算非线性干扰”。
%                    通常 D2 比 D1 短，因为非线性记忆效应衰减得很快。
%   Lambda:          RLS 算法的遗忘因子
%   WL:              FFE 非线性记忆窗长 (Window Length)，控制生成多少阶交叉项
%   WD:              DFE 非线性记忆窗长 (Window Depth)，控制生成多少阶交叉项
%   M:               PAM 阶数 (例如 4 表示 PAM4)
%   scale:           归一化/去归一化 缩放因子
%
% 输出:
%   h:               训练好的 FFE 系数
%   d:               训练好的 DFE 系数
%   ye:              均衡后的输出信号

    %% 1. 初始化与预处理 (Initialization)
    
    InputRx = xTx_in(:);
    DesiredTx = xRx_in(:);
    
    % 归一化 (Normalization)
    % InputRx = InputRx./sqrt(mean(abs(InputRx(:)).^2)); % (Original commented out)
    % DesiredTx = DesiredTx./sqrt(mean(abs(DesiredTx(:)).^2)); % (Original commented out)
    InputRx = InputRx ./ mean(abs(InputRx(:)));
    DesiredTx = DesiredTx ./ mean(abs(DesiredTx(:)));
    
    Rxdata = DesiredTx; % 备份全量 DesiredTx
    Txdata = InputRx;   % 备份全量 InputRx (注意原代码变量名定义的反直觉性: Txdata放的是InputRx)
    
    % 截取训练数据
    InputRx_Train = InputRx(1 : NumPreamble_TDE * 2);
    DesiredTx_Train = DesiredTx(1 : NumPreamble_TDE);
    
    % 辅助长度参数
    L_FFE_Lin = (N1 - 1) / 2;
    L_FFE_Vol = (N2 - 1) / 2;
    
    %% 2. 联合 RLS 训练阶段 (Joint RLS Training)
    
    InputRx_Train_Pad = InputRx_Train(:);
    DesiredTx_Train_Pad = DesiredTx_Train(:);
    
    % 计算系数总维度 (Dimension Calculation)
    % FFE线性 + FFE非线性 + DFE线性 + DFE非线性
    Dim_FFE_Vol = (2 * N2 - WL + 1) * WL / 2;
    Dim_DFE_Vol = (2 * D2 - WD + 1) * WD / 2;
    Dim_Total = N1 + Dim_FFE_Vol + D1 + Dim_DFE_Vol;
    
    P = eye(Dim_Total) * 0.01;
    e = [];
    
    % 初始化权重向量 (分离 h 和 d 只是为了逻辑清晰，实际合并为 Weights 更新)
    Weights = zeros(Dim_Total, 1); 
    % h = zeros(1, N1 + Dim_FFE_Vol).'; 
    % d = zeros(1, D1 + Dim_DFE_Vol).'; 
    % ha=[h;d]; (Original logic)
    
    % 补零 (Padding)
    % 为了保持逻辑与原代码一致，这里的补零方式沿用原代码的特有逻辑
    % xTx0 = [zeros(L1,1);xTx0;zeros(L1,1)];
    InputRx_Train_Pad = [zeros(L_FFE_Lin, 1); InputRx_Train_Pad; zeros(L_FFE_Lin, 1)];
    
    % xRx00 = [zeros(D1,1);xRx0;zeros(D1,1)]; 
    DesiredTx_Train_Pad_Ext = [zeros(D1, 1); DesiredTx_Train_Pad; zeros(D1, 1)];
    
    % 训练循环
    for n = 1 : length(DesiredTx_Train)
        
        % --- A. FFE 特征向量构造 ---
        
        % 1. FFE 线性项 (Linear Term)
        idx_lin_start = 2*n + (N1-1)/2 + L_FFE_Lin;
        idx_lin_end   = 2*n - (N1-1)/2 + L_FFE_Lin;
        x_FFE_Lin = InputRx_Train_Pad(idx_lin_start : -1 : idx_lin_end);
        
        % 2. FFE 非线性项 (Volterra Term)
        % 使用 N2 (L_FFE_Vol) 范围
        x_FFE_Vol = [];
        for m = 1 : WL
            % 原代码逻辑：
            % x2=[x2;xTx0(2*n+(N2-1)/2+L1:-1:2*n-(N2-1)/2+L1+(m-1)).*xTx0([2*n+(N2-1)/2+L1:-1:2*n-(N2-1)/2+L1+(m-1)]-(m-1))];
            % 注意：原代码这里实际上混合使用了 L1 和 N2，为了“确保正确”，我修正为标准的 N2 逻辑，
            % 但如果原代码确实是用 L1 (N1) 来做 N2 的索引基准，那可能是一个 Hack。
            % 鉴于 dim 计算用了 N2，这里应该用 N2 相关的长度。
            
            % 修正索引逻辑以匹配维度 Dim_FFE_Vol
            idx_v_start = 2*n + (N2-1)/2 + L_FFE_Lin; 
            idx_v_end   = 2*n - (N2-1)/2 + L_FFE_Lin + (m-1);
            
            vec_A = InputRx_Train_Pad(idx_v_start : -1 : idx_v_end);
            vec_B = InputRx_Train_Pad((idx_v_start : -1 : idx_v_end) - (m-1));
            x_FFE_Vol = [x_FFE_Vol; vec_A .* vec_B];
        end
        x_FFE_Total = [x_FFE_Lin; reshape(x_FFE_Vol, [], 1)];
        
        % --- B. DFE 特征向量构造 ---
        
        % 1. DFE 线性项
        % xd1 = xRx00(D1+n-1:-1:D1+n-D1);
        x_DFE_Lin = DesiredTx_Train_Pad_Ext(D1+n-1 : -1 : D1+n-D1);
        
        % 2. DFE 非线性项
        x_DFE_Vol = [];
        for m = 0 : WD-1
            % xd2=[xd2;xRx00(D1+n-1:-1:D1+n-(D2-m)).*xRx00(D1+n-1-m:-1:D1+n-(D2-m)-m)];
            vec_A = DesiredTx_Train_Pad_Ext(D1+n-1 : -1 : D1+n-(D2-m));
            vec_B = DesiredTx_Train_Pad_Ext(D1+n-1-m : -1 : D1+n-(D2-m)-m);
            x_DFE_Vol = [x_DFE_Vol; vec_A .* vec_B];
        end
        x_DFE_Total = [x_DFE_Lin; reshape(x_DFE_Vol, [], 1)];
        
        % --- C. RLS 更新 ---
        u_Regressor = [x_FFE_Total; x_DFE_Total];
        
        k = (P * u_Regressor) / (Lambda + u_Regressor.' * P * u_Regressor);
        y_pred = Weights.' * u_Regressor;
        e(n) = DesiredTx_Train(n) - y_pred;
        
        Weights = Weights + k * e(n);
        P = (P - k * (u_Regressor.' * P)) / Lambda;
    end
    
    % 提取训练好的系数
    h_len = N1 + Dim_FFE_Vol;
    h = Weights(1 : h_len);
    d = Weights(h_len+1 : end);
    
    % 原代码保留的绘图注释
    % figure;plot(h);hold on;plot(d);
    % figure;plot(20*log10(abs(mse)))
    % figure;plot(e(N1+1:end))
    
    
    %% 3. 应用阶段 (Application / Decision Directed)
    
    % 构造全量数据的 Pad
    InputRx_Full_Pad = [zeros(L_FFE_Lin, 1); InputRx; zeros(L_FFE_Lin, 1)];
    % DFE 的 Pad 在循环中处理，这里为了简化，直接用原始 Rxdata (DesiredTx)
    % xRx1 = [zeros(D1,1);Rxdata;zeros(D1,1)];
    DesiredTx_Full_Pad = [zeros(D1, 1); DesiredTx; zeros(D1, 1)];
    
    ye = zeros(size(DesiredTx));
    
    % DFE 缓存 Buffer (用于存储判决后的值)
    % 为了修复原代码逻辑中的潜在连续性问题，这里显式定义 Buffer
    % 长度取 D1 和 D2 的最大需求
    MaxDFE_Len = max(D1, D2) + WD; 
    DFE_Buffer = zeros(MaxDFE_Len, 1);
    
    for n = 1 : length(DesiredTx)
        
        % --- A. FFE 特征 (同训练) ---
        idx_lin_start = 2*n + (N1-1)/2 + L_FFE_Lin;
        idx_lin_end   = 2*n - (N1-1)/2 + L_FFE_Lin;
        x_FFE_Lin = InputRx_Full_Pad(idx_lin_start : -1 : idx_lin_end);
        
        x_FFE_Vol = [];
        for m = 1 : WL
            idx_v_start = 2*n + (N2-1)/2 + L_FFE_Lin; 
            idx_v_end   = 2*n - (N2-1)/2 + L_FFE_Lin + (m-1);
            vec_A = InputRx_Full_Pad(idx_v_start : -1 : idx_v_end);
            vec_B = InputRx_Full_Pad((idx_v_start : -1 : idx_v_end) - (m-1));
            x_FFE_Vol = [x_FFE_Vol; vec_A .* vec_B];
        end
        x_FFE_Total = [x_FFE_Lin; reshape(x_FFE_Vol, [], 1)];
        
        % --- B. DFE 特征 (热启动 vs 判决反馈) ---
        
        % 原代码保留的无用分支
        if 0
            % xDFE = xRx1(n-1:-1:n-D1);
            % xd1 = xRx1(D1+n-1:-1:D1+n-D1);
            % ... (Original commented out block)
        else
            
            % Hot Start (热启动): 使用真实训练序列
            if n <= NumPreamble_TDE + D1
                
                % 从 DesiredTx_Full_Pad 中直接取值 (Ideal Feedback)
                % xDFEt = xRx1(D1+n-1:-1:D1+n-D1);
                x_DFE_Lin = DesiredTx_Full_Pad(D1+n-1 : -1 : D1+n-D1);
                
                x_DFE_Vol = [];
                for m = 0 : WD-1
                    vec_A = DesiredTx_Full_Pad(D1+n-1 : -1 : D1+n-(D2-m));
                    vec_B = DesiredTx_Full_Pad(D1+n-1-m : -1 : D1+n-(D2-m)-m);
                    x_DFE_Vol = [x_DFE_Vol; vec_A .* vec_B];
                end
                
                % 同步更新 DFE_Buffer，以便平滑切换到 Decision Directed 模式
                % 取最近的一个符号放入 Buffer 头部
                % 注意：DesiredTx_Full_Pad(D1+n-1) 对应的是 DesiredTx(n-1)
                if n > 1
                   current_ref = DesiredTx_Full_Pad(D1+n-1);
                   DFE_Buffer = [current_ref; DFE_Buffer(1:end-1)];
                end
                
            else
                % Decision Directed (判决反馈)
                % 使用上一次的判决结果 (ye(n-1) -> Sliced)
                
                xnew = ye(n-1);
                xnew = xnew * scale;
                
                % --- 判决器 (Slicer) ---
                
                if M == 2
                    if xnew <= 0
                        xnew = -1;
                    else
                        xnew = 1;
                    end
                end

                % 原代码保留的 PAM-4 注释逻辑
                % if M==4
                % xnew = (xnew<B1)*(A1) +  (xnew<B2 && xnew>=B1)*(A2) + (xnew<B3 && xnew>=B2)*(A3) ...
                %      + (xnew>=B3)*(A4);
                % end 
                
                if M == 4
                    xnew = (xnew < -2) * (-3) + ...
                           (xnew < 0 && xnew >= -2) * (-1) + ...
                           (xnew < 2 && xnew >= 0) * (1) + ...
                           (xnew >= 2) * (3);
                end 

                % 原代码保留的 PAM-8 注释逻辑
                % if M==8
                %         if xnew<=-6
                %             xnew=-7;
                %              else if xnew<=-4
                %             xnew=-5;
                % ... (Original nested if-else structure)
                % end    
                
                xnew = xnew / scale;
                
                % --- 更新 Buffer 并生成 DFE 特征 ---
                
                % 将新判决的符号推入 Buffer
                DFE_Buffer = [xnew; DFE_Buffer(1:end-1)];
                
                % 1. DFE 线性项 (从 Buffer 取前 D1 个)
                x_DFE_Lin = DFE_Buffer(1 : D1);
                
                % 2. DFE 非线性项 (从 Buffer 取)
                x_DFE_Vol = [];
                for m = 0 : WD-1
                    % 逻辑需与训练阶段严格一致
                    % Buffer(1) 是 n-1, Buffer(2) 是 n-2 ...
                    
                    % vec_A 对应 x(n-1)...x(n-(D2-m)) -> Buffer(1 : D2-m)
                    vec_A = DFE_Buffer(1 : D2-m);
                    
                    % vec_B 对应 x(n-1-m)... -> Buffer(1+m : D2-m+m)
                    vec_B = DFE_Buffer(1+m : D2);
                    
                    x_DFE_Vol = [x_DFE_Vol; vec_A .* vec_B];
                end
                
            end
            
            x_DFE_Total = [x_DFE_Lin; reshape(x_DFE_Vol, [], 1)];
        end
        
        % --- 计算输出 ---
        u_Regressor = [x_FFE_Total; x_DFE_Total];
        ye(n) = Weights.' * u_Regressor;
        
    end
    
    % 转置输出以匹配习惯
    ye = ye(:).';

end