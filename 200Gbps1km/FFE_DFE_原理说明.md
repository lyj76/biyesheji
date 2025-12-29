# FFE + DFE 联合均衡器原理详解

本文档详细解释了 `LE_FFE2ps_centerDFE_new.m` 代码背后的算法原理。

## 1. 核心概念：为什么要加 DFE？

### 1.1 FFE 的局限性
前面我们学的 FFE (线性均衡器) 只是简单地把接收到的信号加权求和。
*   **问题**：如果信道衰落非常严重（这在高速光通信或者长距离传输中很常见），或者频谱有深衰落点（Spectral Nulls），FFE 为了强行把凹下去的频谱补平，会同时**放大噪声**。
*   **结果**：信号虽然变平了，但噪声大到把信号淹没了。

### 1.2 DFE (判决反馈均衡器) 的登场
DFE 的思想非常狡猾：**既然我已经判决出了上一个符号是什么（比如是 +3），那我就可以算出这个 +3 对当前符号造成的干扰（后响 ISI），然后直接减掉它！**

它由两部分组成：
1.  **前馈滤波器 (FFE)**: 负责处理**未来**和**当前**的接收信号（消除前响 ISI）。
2.  **反馈滤波器 (DFE)**: 负责处理**过去**已经判决出来的符号（消除后响 ISI）。

**公式变化：**
$$ y(n) = \underbrace{\mathbf{h}^T \mathbf{x}(n)}_{\text{FFE部分}} + \underbrace{\mathbf{d}^T \mathbf{\hat{x}}(n)}_{\text{DFE部分}} $$

*   $\mathbf{h}$: FFE 系数
*   $\mathbf{x}(n)$: 接收信号向量 (Rx)
*   $\mathbf{d}$: DFE 系数
*   $\mathbf{\hat{x}}(n)$: **过去已判决的符号向量** (Decided Symbols)

---

## 2. 算法流程详解

代码将整个过程分为了两个阶段：**联合训练** 和 **应用（判决反馈）**。

### 阶段一：联合 RLS 训练 (Training Phase)

在这个阶段，我们拥有“标准答案”（训练序列）。我们假装这些训练序列就是“完美的判决结果”。

1.  **构造联合向量**：
    我们将 FFE 的输入 $\mathbf{x}_{FFE}$ 和 DFE 的输入 $\mathbf{x}_{DFE}$ 拼成一个超级长的向量。
    $$ \mathbf{x}_{Joint} = \begin{bmatrix} \mathbf{x}_{FFE} \\ \mathbf{x}_{DFE} \end{bmatrix} $$
    *   $\\mathbf{x}_{FFE}$: 来自接收信号 Rx（带噪声、带失真）。
    *   $\\mathbf{x}_{DFE}$: 来自训练序列 Tx（干净、完美）。

2.  **RLS 迭代**：
    使用与纯 FFE 完全相同的 RLS 算法，只是现在的系数向量变长了（包含了 h 和 d）。
    $$ \mathbf{h}_{Joint} = \begin{bmatrix} \mathbf{h} \\ \mathbf{d} \end{bmatrix} $$
    通过迭代，同时算出了最优的前馈系数 $\\mathbf{h}$ 和反馈系数 $\\mathbf{d}$。

### 阶段二：应用/判决模式 (Decision Directed Mode)

这是 DFE 最关键、也是最容易出问题的地方。

1.  **没有标准答案了**：
    训练结束后，我们不再有训练序列。DFE 需要的“过去符号”必须来自**实时的判决结果**。

2.  **判决环路 (Decision Loop)**：
    这是一个连锁反应过程：
    *   **Step 1 滤波**: 用 FFE 处理 Rx，加上 DFE 处理 Buffer 里的旧符号，得到软输出 $y(n)$。
    *   **Step 2 判决 (Slicer)**: 看 $y(n)$ 离哪个标准星座点（如 -3, -1, 1, 3）最近。假设离 2.8 最近，那就判决为 3。
    *   **Step 3 反馈**: 把这个判决结果 "3" 塞进 DFE 的 Buffer 里。
    *   **Step 4 下一轮**: 处理下一个符号时，DFE 就会用到这个 "3" 来计算它对下一个符号的干扰。

3.  **误差扩散 (Error Propagation)**：
    *   如果 Step 2 判决错了（比如把 3 判成了 1），那么 Step 3 塞进 Buffer 的就是错误的 "1"。
    *   这个错误的 "1" 会导致下一个符号的 DFE 计算出错，导致下一个符号也容易判错。
    *   这就是 DFE 的**误码扩散效应**。但只要信噪比还行，DFE 的性能通常远好于 FFE。

---

## 3. 代码中的关键实现细节

### 3.1 联合向量构造 (Joint Regressor)
```matlab
% 代码片段
x_FFE = InputRx_Train_Pad(idx_start : -1 : idx_end);         % FFE: 取 Rx
x_DFE = DesiredTx_Train_Pad(n-1 : -1 : n-FilterLen_DFE);     % DFE: 取 Tx (训练时)
x_Joint = [x_FFE; x_DFE];                                    % 拼起来
```

### 3.2 判决逻辑 (Slicer)
代码中包含了一大段 `if M==4 ...` 的逻辑，这就是**判决器**。
它把模拟电压值（如 2.8V, -0.9V）强行变成数字符号（3, -1）。

```matlab
% 4-PAM 判决逻辑示例
if soft_sym < -2
    hard_sym = -3;
elseif soft_sym < 0
    hard_sym = -1;
...
```

### 3.3 缓冲区更新 (Buffer Shift)
```matlab
% 把最新的判决结果放在最上面，把最老的挤出去
xDFE_Buffer = [hard_sym_final; xDFE_Buffer(1:end-1)];
```
这实现了 DFE 的“记忆”功能，始终记住最近的 `D1` 个符号。

---

## 4. 总结：黑盒视角

如果把这个 `LE_FFE2ps_centerDFE_new.m` 看作黑盒：

*   **输入**：
    1.  `xTx` (Rx数据): 2倍过采样，很脏。
    2.  `xRx` (Tx数据): 训练序列，很干净。
*   **训练过程**：
    *   把 Rx 喂给 FFE，把 Tx 喂给 DFE。
    *   用 RLS 算出怎么加权混合这两者，能得到最完美的输出。
*   **工作过程**：
    *   FFE 继续吃 Rx。
    *   DFE 此时没有 Tx 吃，只能吃**上一步自己吐出来的判决结果**。
    *   两者混合，输出最终净化后的信号。

```