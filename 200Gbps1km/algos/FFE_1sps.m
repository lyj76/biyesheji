function [ye, idxTx] = FFE_1sps(xRx_1sps, xTx_1sps, NumPreamble, TapLen, Lambda)
% FFE_1sps: Standard Symbol-Spaced FFE (T-spaced)
% 
% Inputs:
%   xRx_1sps: Received signal, downsampled to 1 sample/symbol (Phase already selected)
%   xTx_1sps: Training symbols (1 sps)
%   NumPreamble: Number of training symbols
%   TapLen: Equalizer length (must be odd, e.g., 31)
%   Lambda: RLS forgetting factor (e.g., 0.999)
%
% Outputs:
%   ye: Equalized output (1 sps)
%   idxTx: Indices of xTx corresponding to ye (for BER calculation)

    if nargin < 4 || isempty(TapLen), TapLen = 31; end
    if nargin < 5 || isempty(Lambda), Lambda = 0.999; end

    % Ensure column vectors
    rx = xRx_1sps(:);
    tx = xTx_1sps(:);

    % Normalize input
    rx = (rx - mean(rx)) / std(rx);
    % Normalize target
    tx_mean = mean(tx); 
    tx_std = std(tx);
    tx_n = (tx - tx_mean) / tx_std;

    N = length(rx);
    Half = floor(TapLen/2);
    
    % RLS Initialization
    P = eye(TapLen) * 10;
    w = zeros(TapLen, 1);
    
    % Prepare Output
    ye = zeros(N, 1);
    
    % --- Step 1: Alignment (Scan Delay) ---
    % Since it is T-spaced, we only need to find integer delay shift.
    % We use a simple correlation/least-squares scan on the first 1000 samples.
    
    best_mse = inf;
    best_d = 0;
    
    ScanRange = -30:30;
    TrainLen_Scan = min(1000, NumPreamble);
    
    X_scan = zeros(TrainLen_Scan, TapLen);
    % Construct matrix once for delay=0 (base structure)
    % To be safe, start from index 100 to allow TapLen history
    start_offset = 100;
    for i = 1:TrainLen_Scan
        idx = (i + start_offset) + (-Half:Half); 
        % Check bounds
        if idx(end) > length(rx)
            X_scan = X_scan(1:i-1, :);
            break; 
        end
        % Note: RLS often uses u = rx(n:-1:n-L+1), but here we use centered window rx(n-Half:n+Half)
        % For standard FIR: y[n] = w' * x[n]
        % Let's use standard order: [r(n-H) ... r(n+H)]'
        X_scan(i, :) = rx(idx); 
    end
    
    % Scan Delays
    for d = ScanRange
        % Target: tx_n corresponds to center of rx window
        % If delay=0, rx(start_offset+1) aligns with tx(start_offset+1)
        tx_indices = (1:size(X_scan,1)) + start_offset + d;
        
        if any(tx_indices < 1) || any(tx_indices > length(tx_n)), continue; end
        
        y_target = tx_n(tx_indices);
        
        % Fast LS
        w_tmp = X_scan \ y_target;
        y_est = X_scan * w_tmp;
        mse = mean((y_est - y_target).^2);
        
        if mse < best_mse
            best_mse = mse;
            best_d = d;
        end
    end
    
    delay = best_d;
    % disp(['    [FFE-1sps] Best Delay: ', num2str(delay), ', MSE: ', num2str(best_mse)]);
    
    % --- Step 2: RLS Training ---
    start_idx = Half + 1 + start_offset; 
    end_train = start_idx + NumPreamble - 1;
    
    % Reset Weights for real training
    P = eye(TapLen) * 10;
    w = zeros(TapLen, 1);

    for n = start_idx : end_train
        if n > length(rx) - Half, break; end
        
        % Regressor (Column vector)
        u = rx(n-Half : n+Half); 
        
        % Desired
        tx_idx = n + delay;
        if tx_idx < 1 || tx_idx > length(tx_n), continue; end
        d_val = tx_n(tx_idx);
        
        % RLS Update
        y_val = w' * u;
        e = d_val - y_val;
        
        k = (P * u) / (Lambda + u' * P * u);
        w = w + k * e;
        P = (P - k * u' * P) / Lambda;
        
        ye(n) = y_val;
    end
    
    % --- Step 3: Apply to Whole Sequence ---
    % We can use 'filter' function, but manual loop is safer for matching RLS structure
    % Or simply convolution since w is fixed now
    % w is [w(-H) ... w(H)]' corresponding to [r(n-H) ... r(n+H)]'
    % effectively conv(rx, flip(w))
    
    ye_full = conv(rx, flipud(w), 'same');
    
    % Use the RLS trained output for the training part? 
    % Usually for BER evaluation, we use the final weights for the whole sequence (Static Equalization)
    % or we keep updating (Adaptive). Let's use Static (Final Weights) for fair comparison with NN.
    ye = ye_full;

    % De-normalize
    ye = ye * tx_std + tx_mean;
    
    % Output Indices
    idxTx = (1:N)' + delay;

end