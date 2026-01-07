%% RLS 算法验证与测试台 (System Identification Testbench)
% 作者: Gemini & User (Math PhD)
% 目的: 验证 FFE_2pscenter.m 的收敛性与准确性
% 原理: 构造已知信道 h_true，生成对应数据，看算法能否反解出 h_true

%% 使用实测 pam4 数据 (rop3dBm_1.mat, rop5dBm_1.mat) 的快速测试脚本
% 运行 loadpam4_1kmFFE_DFE 并保存结果

clear; close all; clc;

baseDir = fileparts(mfilename('fullpath'));

results = loadpam4_1kmFFE_DFE({fullfile(baseDir,'rop3dBm_1.mat'), ...
                               fullfile(baseDir,'rop5dBm_1.mat')});

disp('测试完成，结果已返回到变量 results。');
