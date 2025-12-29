function [d0, best_ser] = align_symbol_delay_by_ser(ye, xsym, NumPreamble_TDE, M, dCandidates)
% align_symbol_delay_by_ser: scan symbol delay by SER on training segment

if nargin < 5 || isempty(dCandidates)
    dCandidates = -40:40;
end

ye = ye(:);
xsym = xsym(:);

Ntr = min(NumPreamble_TDE, length(ye));
if Ntr < 1
    d0 = 0;
    return;
end

[~, C] = kmeans(double(ye(1:Ntr)), M, 'Replicates', 3);
levels = sort(C(:)).';
thr = (levels(1:M-1) + levels(2:M)) / 2;

yq = slice_levels(ye, levels, thr);
ysym = pamdemod(yq, M, 0, 'gray');

best_ser = inf;
d0 = 0;
for d = dCandidates
    idxTx = (1:length(ysym)).' + d;
    m = idxTx >= 1 & idxTx <= NumPreamble_TDE;
    if nnz(m) < 2000
        continue;
    end
    ser = mean(ysym(m) ~= xsym(idxTx(m)));
    if ser < best_ser
        best_ser = ser;
        d0 = d;
    end
end
end

function yq = slice_levels(y, levels, thr)
y = double(y);
yq = zeros(size(y));
yq(y < thr(1)) = levels(1);
for k = 2:length(thr)
    yq(y >= thr(k-1) & y < thr(k)) = levels(k);
end
yq(y >= thr(end)) = levels(end);
end
