function [best_offset, best_d0] = align_offset_delay_by_ser(y, xsym, NumPreamble_TDE, M, dCandidates)
% align_offset_delay_by_ser: scan offset (1/2) and symbol delay by SER

if nargin < 5 || isempty(dCandidates)
    dCandidates = -40:40;
end

y = y(:);
xsym = xsym(:);
N = length(y);

% crude 2sps detection
is2sps = N > 1.5 * length(xsym);
offsetCandidates = 1;
if is2sps
    offsetCandidates = [1 2];
end

best_ser = inf;
best_offset = 1;
best_d0 = 0;

for off = offsetCandidates
    if is2sps
        y1 = y(off:2:end);
    else
        y1 = y;
    end

    Ntr = min(NumPreamble_TDE, length(y1));
    if Ntr < 1
        continue;
    end

    [~, C] = kmeans(double(y1(1:Ntr)), M, 'Replicates', 3);
    levels = sort(C(:)).';
    thr = (levels(1:M-1) + levels(2:M)) / 2;

    yq = slice_levels(y1, levels, thr);
    ysym = pamdemod(yq, M, 0, 'gray');

    for d0 = dCandidates
        idxTx = (1:length(ysym)).' + d0;
        m = idxTx >= 1 & idxTx <= NumPreamble_TDE;
        if nnz(m) < 2000
            continue;
        end
        ser = mean(ysym(m) ~= xsym(idxTx(m)));
        if ser < best_ser
            best_ser = ser;
            best_offset = off;
            best_d0 = d0;
        end
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
