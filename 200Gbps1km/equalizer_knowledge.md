# Load 程序通用处理与 7 种均衡算法差异（200Gbps / 1 km / PAM4）最终笔记

这份笔记总结：为什么同一条主链路下 7 个算法会表现差异巨大，以及如何在 `loadpam4_1kmFFE_DFE.m` 里做到**可插拔 + 可比对 + 稳健评估**（对齐、判决、BER/SNR）。

---

## 0. 结论先行：要“稳健”，必须统一两件事

所有算法不管内部怎么做，最后必须被统一成：

1. **索引域统一（谁对应谁）**
   把均衡器输出映射到发送符号索引域：

* `out.ye`：有效输出序列（1 sps 的符号序列）
* `out.idxTx`：`out.ye(i)` 对应的 Tx 符号索引

2. **判决域统一（幅度尺度）**
   传统均衡器输出通常满足：`ye ≈ g*xm + c`（任意增益/偏置）。
   因此判决前必须做训练段仿射标定：

* 用训练段拟合 `xm ≈ a*ye + b`
* 测试段用 `xhat = a*ye + b` 再 `pamdemod`（Gray）判决

> 如果只做了对齐但不做标定，最典型现象是：**MSE 很小、SNR 很高，但 BER 卡在 ~0.25**。

---

## 1. 关键文件（最终结构）

* `loadpam4_1kmFFE_DFE.m`：主流程 + 算法切换 + 统一评估入口
* `align_offset_delay_by_ser.m`：**传统算法**对齐（扫 offset + delay）
* `eval_equalizer_pam4.m`：统一评估（**仿射标定 + PAM4 判决 + BER/SER/SNR**）
* 各算法实现：

  * `FFE_2pscenter.m`
  * `VNLE2_2pscenter.m`
  * `LE_FFE2ps_centerDFE_new.m`
  * `DP_VFFE2pscenter_VDFE.m`
  * `CLUT_VDFE_Implementation.m`
  * `FNN_Implementation.m`
  * `RNN_Implementation.m`

---

## 2. load 主流程（统一入口）

主流程本质是：**固定链路 + 可插拔均衡器 + 统一评估**。

1. **Tx 生成与成形**

* `PAMSource(M, NumSymbols)` → `xsym`（符号索引）+ `xm`（pammod 幅度）
* `xs = xm`
* 上采样（2 sps） + Raised Cosine 成形得到 `x_shape`

2. **加载实验数据与同步**

* load `ReData`，取反：`ReData = -ReData`
* `TEFEMMM2` 做定时估计 `TE`
* 取对齐段：`ysync = ReData(1024 + 20 + TE : end)` 并截到 `length(x_shape)`

3. **接收端匹配滤波**

* 去除边缘：`yt_filter = ysync(1+N/2 : end-N/2)`
* `xRx = yt_filter`（2 sps）
* `xTx = xs`（1 sps）
* `NumPreamble_TDE = 10000`（训练段符号数）

4. **算法切换（7 选 1）**
   只需要切换均衡器调用，其余流程不变。

5. **统一输出映射（最关键）**
   生成 `out.ye` + `out.idxTx`

6. **统一评估**
   `eval_equalizer_pam4(out.ye, out.idxTx, xsym, xm, NumPreamble_TDE, M)`
   输出 BER/SER/SNR，保证所有算法可比。

---

## 3. 7 种算法“本质差异”对比（模型 / 输入 / 训练 / 输出语义）

| 算法           | 核心结构                        | 输入特征                                        | 训练/更新                  | 输出语义（关键）                            |
| ------------ | --------------------------- | ------------------------------------------- | ---------------------- | ----------------------------------- |
| FFE          | 线性 FIR + RLS                | Rx(2sps) 的局部窗（fractionally spaced / center） | RLS 逐符号                | 输出 `ye`：可能已是 1sps；也可能仍含 2sps/裁边/群时延 |
| VNLE         | Volterra FFE + RLS          | 线性 N1 + 非线性 N2/WL                           | RLS                    | `ye` 语义同上，但非线性更强                    |
| LE_FFE+DFE   | 线性 FFE + 线性 DFE             | Rx 窗口 + 过去判决                                | RLS + 判决反馈             | `ye` 同上，且可能受误差传播影响                  |
| DP_VFFE+VDFE | Volterra FFE + Volterra DFE | N1/N2/WL + D1/D2/WD                         | 联合 RLS + 反馈            | `ye` 同上，误差传播更显著                     |
| CLUT-VDFE    | DFE 系数聚类→查表                 | LUT 替代部分乘法                                  | 训练→聚类→运行查表             | 输出可能全长也可能有效段，仍建议统一对齐/评估             |
| FNN          | 前馈 MLP                      | 2sps 中心窗（InputLength）                       | Adam（batch）            | 返回 `ye_valid` + `valid_idx`（Tx 域索引） |
| AR-RNN       | 显式输出反馈 + MLP（或等效）           | 中心窗 + k 个历史输出                               | Adam + free-running 推理 | 返回 `ye` + `valid_idx`（或内部定义的 Tx 映射） |

> 注意：NN “不需要对齐”的说法不准确。NN 的对齐发生在**数据集构造阶段**（best_offset / best_delay），最后用 `valid_idx` 固化映射。

---

## 4. 通用化的核心：统一输出映射 `out.ye + out.idxTx`

### 4.1 传统均衡器（FFE/VNLE/DFE/DP/CLUT…）

问题：`ye` 可能存在

* 2 sps / 1 sps 不确定
* 群时延（符号级 delay）
* 头尾裁剪或有效段缩短

做法：**训练段扫 offset + delay**（用 SER 指标）

* `align_offset_delay_by_ser(ye, xsym, NumPreamble_TDE, M, dCandidates)`

  * `offset`：如果 `ye` 像 2sps，则在 {1,2} 扫；否则 offset=1
  * `delay`：扫 `-60:60`（符号级）

得到最佳 `(off, d0)` 后：

* 若 `ye` 是 2sps：`ye_use = ye(off:2:end)` → 1sps
* 否则：`ye_use = ye(:)`
* `idxTx = (1:length(ye_use)).' + d0`

### 4.2 NN 均衡器（FNN / AR-RNN）

* 直接使用输出的 `valid_idx`（Tx 域索引）
* `idxTx = valid_idx`
* `ye_use = ye_valid` 或 `ye(valid_idx)`（取决于你实现返回的形式）

> **强制规则：主程序不要用 `exist('valid_idx')` 去猜算法类型**。必须用显式 `algo_id` / `switch`，并且每次调用算法前要 `clear` 防止变量残留污染。

---

## 5. 通用判决与 BER 评估：`eval_equalizer_pam4`（最终正确做法）

### 5.1 为什么不能直接 `pamdemod(ye)`

`pamdemod(x, M, 0, 'gray')` 默认假设输入在标准 PAM 星座幅度域。
但均衡器输出通常 `ye = g*xm + c`（任意增益/偏置），不标定就会出现：

* SNR 看起来很高
* BER 却接近随机（常见 ~0.25）

### 5.2 正确统一判决：训练段仿射标定

在训练段拟合：

* `xm ≈ a*ye + b`

用最小二乘得到 `a,b`，然后对测试段：

* `xhat = a*ye_test + b`
* `ysym = pamdemod(xhat, M, 0, 'gray')`
* 与 `xsym(idx_test)` 计算 BER/SER

SNR 也用同一个标定后的误差定义，保证可比。

> 这个“线性校准”是**判决域标定**，不是信道逆建模。目标是把输出拉回 pammod 的幅度域，让所有算法在同一判决规则下对比。

---

## 6. BER 差异来自哪里（你现在看到的现象都合理）

1. **对齐误差（offset/delay）**

* 传统算法靠扫描对齐；扫错 → BER 直接崩
* NN 的对齐依赖数据构造阶段选的 offset/delay；选错一样崩

2. **反馈结构的误差传播**

* DFE / DP-VDFE / AR-RNN free-running 都可能出现 error propagation
* teacher forcing（训练用真值反馈）只能缓解初期，不保证推理完全稳定

3. **窗口长度与非线性记忆**

* Volterra / DP-VDFE 对非线性记忆覆盖更强，通常能打更低 BER
* NN 要足够容量 + 足够训练样本 + 正确对齐，否则会输给 Volterra/RLS

4. **评估层的标定**

* 不做仿射标定 → 你会看到“误码 0.25”的假随机
* 做了标定 → 才能比较算法本身能力

---

## 7. 最终稳健性 Checklist（工程必备）

**每次切换算法前：**

* `clear ye ye_valid valid_idx net out stats idxTx ye_use d0 off`

**输出统一要求：**

* 必须生成 `out.ye`（1sps 有效输出）和 `out.idxTx`（Tx 索引）
* BER/SNR 只能在 `out.idxTx > NumPreamble_TDE` 的测试段统计

**判决统一要求：**

* 一律用 `eval_equalizer_pam4` 的仿射标定后再 `pamdemod`
* 不要再用 `Normalizepam` 来“碰运气”对齐尺度（那会掩盖真实差异）

**快速定位异常：**

* **MSE 小 / SNR 高 / BER≈0.25** → 先查判决标定（是否用了 `a*ye+b`）
* **BER≈0.5** → 先查对齐/抽样（offset/delay / idxTx 是否错）
* **NN 偶尔一条数据崩** → 多半是 free-running 误差传播 + 对齐边界不稳（可用硬判决反馈、热启动、缩短 k、或改推理策略）

---

## 8. 扩展建议（以后想进一步提升一致性/性能）

* 在 `eval_equalizer_pam4` 里增加可选的更强标定：

  * 训练段拟合二次多项式标定（只在输出明显非仿射时用）
  * 或直接用训练段估计 3 个阈值（decision boundary calibration）
* 对 AR-RNN：推理阶段可选 “soft→hard 混合反馈” 逐步过渡，降低误差传播

---

### 你现在这套“通用化处理”算合理吗？

算，而且是做系统对比最工程化、最稳的方式之一：
**对齐统一 + 标定统一**之后，算法行为再复杂也能可比，差异主要来自算法本身（窗口、非线性、反馈传播、训练收敛），而不是统计误差。
