%% Test Correlation Significance
clear; close all;
addpath('fns');

M = 4;
N = 10000;
Osamp = 2;

% Sequence A
s1 = RandStream.create('mt19937ar', 'seed', 1);
RandStream.setGlobalStream(s1);
[~, xA] = PAMSource(M, N);
xA_up = rectpulse(xA, Osamp);

% Sequence B (Different Seed)
s2 = RandStream.create('mt19937ar', 'seed', 2);
RandStream.setGlobalStream(s2);
[~, xB] = PAMSource(M, N);
xB_up = rectpulse(xB, Osamp);

% Cross Corr
[c, ~] = xcorr(abs(xA_up), abs(xB_up), 'coeff');
max_c = max(c);

disp(['Max Corr between UNRELATED sequences (Abs): ', num2str(max_c)]);
