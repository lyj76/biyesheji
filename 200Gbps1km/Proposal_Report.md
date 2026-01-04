# 毕业设计（论文）开题报告

题目：面向200Gbps短距光互连的过采样RNN与CLUT协同非线性均衡算法研究

学生姓名：罗永健

学号：22309106

专业：电子信息工程通信工程

指导教师：张俊威

填表日期：2026年1月2日

## 一、 国内外关于本选题的研究现状、水平和发展趋势，选题研究的目的和意义

### 1. 选题研究的目的和意义

**研究目的：**
本课题旨在针对下一代单波200Gbps短距光互连系统（Short-Reach Optical Interconnects）中极其严峻的带宽限制和非线性损伤问题，研究并提出一种基于**2倍过采样（2 SPS）架构的递归神经网络（RNN）与聚类查找表（CLUT）协同非线性均衡算法**。通过结合RNN对时序记忆效应的强大建模能力与CLUT的低复杂度查表特性，旨在突破传统线性均衡器的性能瓶颈，并在计算复杂度与传输性能之间取得最优平衡，为未来高速数据中心光互连提供具有工程实用价值的DSP解决方案。

**研究意义：**
随着云计算、物联网（IoT）和人工智能（AI）大模型的爆发式增长，全球数据中心内部流量正以每年25%以上的速度激增。光互连技术作为数据中心网络的“血管”，其单波速率正加速从100Gbps向200Gbps乃至400Gbps演进。
1.  **克服物理瓶颈**：在200Gbps超高速率下，低成本光电器件（如VCSEL、DML）的带宽限制和器件非线性（如频率啁啾、饱和效应）成为限制传输距离的主要因素。传统的线性前馈均衡器（FFE）已无法满足IEEE 802.3及OIF标准对误码率（BER）的要求。
2.  **推动算法创新**：现有的Volterra级数均衡器（VNLE）虽然性能优异但计算复杂度过高，难以硬件实现；而常规前馈神经网络（FNN）对信道记忆效应处理能力不足。本课题探索的RNN与CLUT协同架构，符合“以算力换带宽”的技术发展趋势，具有重要的学术价值。
3.  **工程实用价值**：引入2倍过采样机制，能有效抵抗采样时钟相位抖动（Clock Jitter）和信号混叠，显著提升系统的鲁棒性，这对低成本、非制冷的光收发模块实际部署至关重要。

### 2. 国内外研究现状与发展趋势

目前，针对高速强度调制/直接检测（IM/DD）系统的信号均衡技术，国内外研究主要集中在以下三个方向：

**(1) 传统线性与非线性均衡技术的演进**
传统的线性均衡器（FFE/DFE）因结构简单、功耗低而被广泛应用，但在单波速率超过100Gbps且信道带宽受限严重时，其性能急剧下降。
为了解决非线性损伤，**Volterra非线性均衡器（VNLE）**被公认为高性能的基准方案。然而，VNLE的计算复杂度随阶数和记忆长度呈几何级数增长，导致其在ASIC芯片中实现极其困难。
*   **发展趋势**：从全结构VNLE向**稀疏化和降维化**发展。例如，**CLUT（Cluster-Look-Up-Table，聚类查找表）**技术通过K-means聚类算法将接收信号映射到查找表，利用查表操作替代繁琐的乘法运算，极大地降低了硬件实现难度。Zhang Junwei等人（Optics Letters, 2021）提出的CLUT-VDFE算法在保持VNLE性能的同时，将乘法器数量降低了90%以上。

**(2) 基于神经网络（NN）的智能均衡算法**
随着深度学习的发展，神经网络因其强大的非线性拟合能力被引入光通信DSP。
*   **前馈神经网络（FNN）**：早期研究多采用FNN，但其缺乏内部反馈机制，处理具有强记忆效应的光纤色散损伤时效率较低，需要巨大的输入层规模。
*   **递归神经网络（RNN）**：由于引入了隐藏层的反馈回路，RNN天然适合处理时间序列数据，能有效通过内部状态捕捉信道的记忆特征。国内外多项研究（如墨尔本大学Xu Zhaopeng等，JLT 2021）表明，在相同计算复杂度下，RNN（特别是Reservoir Computing和GRU/LSTM变体）的均衡性能显著优于FNN和VNLE。
*   **发展趋势**：从通用大模型向**轻量化、针对性强**的小型网络演进，结合迁移学习以适应光链路的时变特性。

**(3) 过采样（Fractionally Spaced）技术的应用**
在高速串行接口（SerDes）领域，过采样（如2 samples per symbol, 2 SPS）是抵抗采样相位偏差的标准技术。然而，在光通信非线性均衡算法中，目前大多数研究仍基于波特率采样（1 SPS）。
*   **研究缺口**：1 SPS架构对时钟恢复电路（CDR）的精度要求极高，任何采样相位的漂移都会导致性能雪崩。将2 SPS架构引入RNN或CLUT非线性均衡器，虽然会增加数据吞吐量，但能显著提升系统对时序误差的容忍度，是目前高鲁棒性DSP算法的重要研究方向。

**总结：** 综上所述，单波200Gbps PAM4光传输系统正面临带宽与非线性的双重挑战。现有的单一算法难以同时满足高性能、低复杂度和高鲁棒性的要求。将**RNN的高维特征提取能力**与**CLUT的快速查表能力**相结合，并置于**2倍过采样**架构下，是符合光通信DSP发展规律的前沿探索。

## 二、 选题研究的计划进度及可行性论述

###  1. 计划进度
第1-2周：查阅文献，精读参考论文（JLT 2021 Xu et al., Optics Letters Zhang et al.），理解RNN和CLUT的数学原理。

第3-4周：完成200Gbps PAM4系统仿真链路搭建，实现基础的FFE和DFE算法作为基准。

第5-7周：（已完成基础）编写并调试RNN和CLUT核心代码，实现 比较程序 框架，完成初步的数据跑通。

第8-10周：针对2 SPS输入特性优化算法参数（如RNN的反馈长度k、CLUT的聚类中心数）；进行大规模数据测试（不同ROP）。

第11-12周：整理实验数据，绘制BER曲线、星座图；撰写毕业论文。

第13周：论文修改、查重与答辩准备。

### 2. 可行性论述
(1) 已充分调研了国内外相关文献，对RNN和Volterra级数原理有深入理解。

(2) 目前已完成了核心代码 roll_7.m 的编写，实现了包括FFE, VNLE, LE-FFE-DFE, DP-VFFE-VDFE, CLUT-VDFE, FNN, RNN在内的7种算法。初步测试显示RNN在2 SPS输入下具有优异的抗噪性能。

(3) 实验条件具备：拥有完整的MATLAB仿真环境和200Gbps 1km传输实验数据，能够支撑算法的验证与迭代。

## 三、 毕业论文（设计）撰写提纲

### 第一章 绪论

1.1 研究背景与意义

1.2 短距IM/DD光通信系统的主要损伤（带宽限制、器件非线性）

1.3 非线性均衡技术研究现状

1.4 本文主要工作与创新点（2 SPS架构、RNN与CLUT结合）

### 第二章 200Gbps PAM4光传输系统与信道均衡理论

2.1 系统模型与信号产生

2.2 信道损伤数学模型

2.3 线性均衡原理（FFE与DFE）

2.4 过采样（Fractionally Spaced）均衡的优势分析

### 第三章 基于过采样的低复杂度CLUT-VDFE算法设计

3.1 Volterra级数原理及其复杂度瓶颈

3.2 CLUT（聚类查找表）算法原理

3.3 2 SPS CLUT-VDFE的具体实现与参数优化

3.4 仿真验证与性能分析

### 第四章 基于过采样的高性能AR-RNN均衡算法设计

4.1 递归神经网络（RNN）与长短期记忆

4.2 AR-RNN（自回归RNN）结构设计

4.3 训练策略：Dropout正则化防止过拟合

4.4 2 SPS输入下的特征增强与网络优化

4.5 实验结果：与线性DFE及FNN的对比

### 第五章 RNN与CLUT的性能对比与协同架构探讨

5.1 不同ROP下的误码率（BER）对比分析

5.2 计算复杂度与收敛性对比

5.3 协同方案初探：RNN处理残差的潜力

5.4 2 SPS相对于1 SPS的性能增益量化

### 第六章 总结与展望

6.1 全文总结

6.2 存在的问题与未来工作方向

## 四、 参考文献

[1] J. Zhang, J. Yu, and N. Chi, "Short-Reach Optical Interconnects: Status and Prospects," *Journal of Lightwave Technology*, vol. 38, no. 18, pp. 4991-5006, 2020. (综述：短距光互连现状)
[2] S. Zhang *et al.*, "Beyond 100 Gb/s Electrical Interface: A Survey of Standards and Technologies," *IEEE Access*, vol. 8, pp. 129267-129278, 2020. (综述：接口标准演进)
[3] J. Zhang, H. Tan, A. P. T. Lau, Z. Li, and C. Lu, "Low-complexity cluster-assisting look-up-table-based Volterra decision-feedback equalizer for IM/DD systems," *Optics Letters*, vol. 46, no. 5, pp. 1013-1016, 2021. (核心参考：CLUT算法)
[4] Z. Xu *et al.*, "High-Speed Short-Reach Optical Communications With Computational Efficiency: A Review of Neural Network Equalization," *Journal of Lightwave Technology*, vol. 39, no. 4, pp. 915-928, 2021. (核心参考：神经网络均衡综述)
[5] C. Ye *et al.*, "Recurrent Neural Network Based Nonlinear Equalization for High-Speed PAM-4 Transmission," *IEEE Photonics Technology Letters*, vol. 30, no. 14, pp. 1305-1308, 2018. (RNN经典论文)
[6] T. Wettlin *et al.*, "Complexity Reduction of Volterra Nonlinear Equalization for Optical Short-Reach IM/DD Systems," *Journal of Lightwave Technology*, vol. 38, no. 18, pp. 5057-5065, 2020. (Volterra复杂度降低)
[7] X. Li *et al.*, "200 Gb/s/lane PAM-4 IM/DD transmission over 10 km SSMF using a low-complexity nonlinear equalizer," *Optics Express*, vol. 31, no. 2, pp. 2567-2578, 2023. (200G最新进展)
[8] K. Zhong *et al.*, "1.6 Tb/s (8× 200 Gb/s) PAM-4 Transmission over 2 km SSMF Using a Single-Chip Coherent Transceiver," *Journal of Lightwave Technology*, vol. 39, no. 4, pp. 929-936, 2021. (200G实验验证)
[9] F. Buchali *et al.*, "Rate Adaptive Optical Transmission Using Variable Subcarrier Spacing and Fractionally Spaced Equalization," *Journal of Lightwave Technology*, vol. 40, no. 11, pp. 3456-3464, 2022. (过采样均衡)
[10] D. Wang *et al.*, "Nonlinear Equalization for 200-Gb/s PAM-4 Short-Reach Optical Interconnects Using Reservoir Computing," in *Optical Fiber Communication Conference (OFC)*, 2020, Paper T3E.4. (储备池计算)
[11] A. Dochhan *et al.*, "Fractionally Spaced Equalization for High-Speed Optical Interconnects," *IEEE Photonics Journal*, vol. 11, no. 2, pp. 1-10, 2019. (FSE原理)
[12] K. Zhang *et al.*, "Low-complexity nonlinear equalizer based on field-programmable gate array for short-reach optical interconnects," *Optical Engineering*, vol. 57, no. 4, 2018.

