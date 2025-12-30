function stats = eval_equalizer_pam4(ye, idxTx, xsym, xm, NumPreamble_TDE, M)
% ye: continuous output (valid samples only)
% idxTx: Tx symbol indices for ye (same length)
% xsym: transmitted symbol indices (0..M-1)
% xm: transmitted PAM amplitudes (pammod output)
% return: BER/SER/SNRdB + levels/thr

idxTx = idxTx(:);
ye = ye(:);

% ensure equal length to avoid size mismatch
if length(idxTx) ~= length(ye)
    n = min(length(idxTx), length(ye));
    idxTx = idxTx(1:n);
    ye = ye(1:n);
end

% keep only valid indices/samples
m = idxTx >= 1 & idxTx <= length(xsym) & isfinite(ye);
idxTx = idxTx(m);
ye = ye(m);

idx_train = idxTx(idxTx <= NumPreamble_TDE);
idx_test  = idxTx(idxTx >  NumPreamble_TDE);

if isempty(idx_test) || isempty(idx_train)
    warning('No train/test samples. Check alignment / NumPreamble_TDE.');
    stats.BER = NaN; stats.SER = NaN; stats.SNRdB = NaN;
    stats.levels = []; stats.thr = [];
    return;
end

    y_train = ye(idxTx <= NumPreamble_TDE);
    y_test  = ye(idxTx >  NumPreamble_TDE);

    y_train = y_train(:); % Force column
    xm_train = xm(idx_train);
    xm_train = xm_train(:); % Force column
    
    xm_test = xm(idx_test);
    xm_test = xm_test(:);

    A = [double(y_train), ones(length(y_train),1)];
    p = A \ double(xm_train);a = p(1);
b = p(2);

xhat = a * double(y_test) + b;
ysym_test = pamdemod(xhat, M, 0, 'gray');
xsym_test = xsym(idx_test);

[~, ber] = biterr(ysym_test, xsym_test, log2(M));
[~, ser] = symerr(ysym_test, xsym_test);

% SNR using same calibration
err = xhat - double(xm_test);
snr_lin = mean(abs(double(xm_test)).^2) / (mean(abs(err).^2) + eps);
snr_db = 10 * log10(snr_lin);

stats.BER = ber;
stats.SER = ser;
stats.SNRdB = snr_db;
stats.levels = [];
stats.thr = [];
stats.cal_a = a;
stats.cal_b = b;
end
