function stats = safe_eval_equalizer_pam4(ye, idxTx, xsym, xm, NumPreamble_TDE, M)
% 防止 idx 越界、长度不一致、空向量导致 eval_equalizer_pam4 报错

    ye = ye(:);
    idxTx = idxTx(:);

    L = min(length(ye), length(idxTx));
    ye = ye(1:L);
    idxTx = idxTx(1:L);

    % 去掉越界索引
    good = (idxTx >= 1) & (idxTx <= length(xsym)) & isfinite(ye);
    ye = ye(good);
    idxTx = idxTx(good);

    if isempty(ye) || isempty(idxTx)
        stats.BER = NaN;
        stats.SER = NaN;
        stats.SNRdB = NaN;
        warning('safe_eval_equalizer_pam4: empty valid samples.');
        return;
    end

    stats = eval_equalizer_pam4(ye, idxTx, xsym, xm, NumPreamble_TDE, M);
end