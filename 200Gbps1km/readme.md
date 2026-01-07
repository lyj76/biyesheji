你要“7 个算法都完全稳健”，核心就一句话：

> **把“均衡器输出 ye”统一映射到 Tx 的符号索引域**（也就是知道 ye 的第 i 个样本对应 Tx 的第几个符号），然后 **只在这个映射上做判决/BER/SNR**。
> 不要再出现“有的算法 ye 是全长、有的算法 ye 只有有效段、有的算法自带群时延”的混用。

下面给你一个能一劳永逸的方案：**统一输出接口 + 统一对齐扫描 + 统一判决统计**。你照着做，7 个算法都不会再被“统计错位/0 填充/Normalizepam 漂移”坑。

---

# 方案：统一成一个结构体 `out`，只用 `out.idxTx` 来算 BER

每个算法跑完都返回（或包装成）：

* `out.ye`：连续幅度（长度 = Nvalid）
* `out.idxTx`：每个 `ye` 样本对应的 Tx 索引（长度 = Nvalid）
* `out.name`：算法名

这样你后面统计完全统一：

* 训练段：`idxTx <= NumPreamble_TDE`
* 测试段：`idxTx > NumPreamble_TDE`
* 参考：`xsym(out.idxTx)` / `xm(out.idxTx)`

**永远不再用 calc_range，也永远不把整条 ye 做 Normalizepam。**

---

# Step 1：写一个统一的“对齐 + 判决 + BER”函数（对所有算法通用）

把下面存成 `eval_equalizer_pam4.m`（一个文件就够）：

```matlab
function stats = eval_equalizer_pam4(ye, idxTx, xsym, xm, NumPreamble_TDE, M)
% ye: 连续输出（只含有效点）
% idxTx: ye 对应的 Tx 符号索引（同长度）
% xsym: 发送符号索引(0..M-1)
% xm: 发送幅度符号（pammod 输出）
% 返回：BER/SER/SNRdB + levels/thr（方便复用）

idxTx = idxTx(:);
ye = ye(:);

% 只保留合法索引
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

% 用训练段估计 4 电平阈值（幅度域，稳健）
[~, C] = kmeans(double(y_train(:)), M, 'Replicates', 5);
levels = sort(C(:)).';
thr = (levels(1:M-1) + levels(2:M)) / 2;

% 硬切片（测试段）
yq = slice_levels(y_test, levels, thr);

ysym_test = pamdemod(yq, M, 0, 'gray');
xsym_test = xsym(idx_test);

[~, ber] = biterr(ysym_test, xsym_test, log2(M));
[~, ser] = symerr(ysym_test, xsym_test);

% 稳健 SNR：最小二乘增益拟合
xm_test = xm(idx_test);
a = (xm_test' * y_test) / (xm_test' * xm_test + eps);
y_fit = a * xm_test;
err = y_test - y_fit;
snr_lin = mean(abs(y_fit).^2) / (mean(abs(err).^2) + eps);
snr_db = 10*log10(snr_lin);

stats.BER = ber;
stats.SER = ser;
stats.SNRdB = snr_db;
stats.levels = levels;
stats.thr = thr;
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
```

> 以后你统计 BER/SNR 只调用这个函数，**不会再乱**。

---

# Step 2：每个算法都要给出 `idxTx`（关键）

## A) FNN / AR-RNN（你已经有）

* FNN/RNN 已经返回 `valid_idx`（这是 Tx 索引域）
* 所以直接：

  * `ye_valid = ye(valid_idx);`
  * `idxTx = valid_idx;`

## B) FFE/VNLE/DFE/DP-VDFE/CLUT-VDFE（通常不返回索引）

这些算法大多只返回 `ye`，但 `ye(i)` 对应 Tx 的哪个符号 **不确定**（群时延、裁剪、内部丢样）。

最稳健的统一方法：

> **给每个算法在输出后做一次“符号级对齐扫描”，找到 ye 相对 Tx 的最优符号延迟 d0，然后构造 idxTx。**

我建议把这个写成一个通用对齐函数：

```matlab
function d0 = align_symbol_delay_by_ser(ye, xsym, NumPreamble_TDE, M, dCandidates)
% 用训练段，在判决域扫描符号延迟 d0，选 SER 最小者

if nargin < 5, dCandidates = -40:40; end

% 先用训练段估计 levels/thr
Ntr = min(NumPreamble_TDE, length(ye));
[~, C] = kmeans(double(ye(1:Ntr)), M, 'Replicates', 3);
levels = sort(C(:)).';
thr = (levels(1:M-1)+levels(2:M))/2;

yq = slice_levels(ye(:), levels, thr);
ysym = pamdemod(yq, M, 0, 'gray');

best_ser = inf; d0 = 0;
for d = dCandidates
    idxTx = (1:length(ysym))' + d;      % ye的第n点对应Tx的 n+d
    m = idxTx>=1 & idxTx<=NumPreamble_TDE;
    if nnz(m) < 2000, continue; end
    ser = mean(ysym(m) ~= xsym(idxTx(m)));
    if ser < best_ser
        best_ser = ser; d0 = d;
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
```

对 FFE/VNLE 等算法你就这样统一：

```matlab
d0 = align_symbol_delay_by_ser(ye, xsym, NumPreamble_TDE, M, -40:40);
idxTx = (1:length(ye))' + d0;
ye_valid = ye(:);
```

然后交给 `eval_equalizer_pam4` 去算。

---

# Step 3：主程序里怎么写（7 算法统一模板）

你每次跑完一个算法后，做：

```matlab
if exist('valid_idx','var') && ~isempty(valid_idx)
    % NN类：valid_idx 本来就是 Tx 索引
    idxTx = valid_idx(:);
    ye_use = ye(idxTx);
else
    % 传统均衡器：先扫 d0 对齐到 Tx
    d0 = align_symbol_delay_by_ser(ye, xsym, NumPreamble_TDE, M, -40:40);
    idxTx = (1:length(ye))' + d0;
    ye_use = ye(:);
end

stats = eval_equalizer_pam4(ye_use, idxTx, xsym, xm, NumPreamble_TDE, M);

disp(['BER = ', num2str(stats.BER), ', SNR(dB)=', num2str(stats.SNRdB)]);
```

⚠️ 注意：对 NN 类我写了 `ye_use = ye(idxTx)` 是因为你的 NN 输出 `ye` 是“全长向量”，但只有 valid_idx 有效；如果你 NN 输出已经是 “只含有效段”，那就直接 `ye_use = ye`。

---

# 你现在最该做的 2 个实用建议

1. **彻底停用 Normalizepam() 做 BER 判决**（它会让不同算法之间对比失真，且对 AR/RNN 会门限漂移）
2. 所有算法输出都走同一个 `eval_equalizer_pam4()`，这样你再也不会出现：

   * FFE 训练 MSE 很好但 BER=0.25
   * RNN 被 0 padding 污染到 0.5

---

如果你愿意，我可以把你现在 `loadpam4_1kmFFE_DFE.m` 的“均衡器选择 + 统计”部分直接改写成一个干净版本（保留你原结构、但统一调用这两个函数），你只要复制粘贴就能跑。
