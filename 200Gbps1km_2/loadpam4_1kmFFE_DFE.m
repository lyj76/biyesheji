function results = loadpam4_1kmFFE_DFE(dataFiles)
%% PAM-4 仿真与实测数据验证入口
% 默认读取 rop3dBm_1.mat 与 rop5dBm_1.mat，生成 BER/SNR 结果并绘图

close all;

baseDir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(baseDir,'fns')));

if nargin < 1 || isempty(dataFiles)
    dataFiles = {fullfile(baseDir,'rop3dBm_1.mat'), fullfile(baseDir,'rop5dBm_1.mat')};
else
    dataFiles = cellstr(dataFiles);
    for idx = 1:numel(dataFiles)
        if ~isfile(dataFiles{idx})
            candidate = fullfile(baseDir, dataFiles{idx});
            if isfile(candidate)
                dataFiles{idx} = candidate;
            end
        end
    end
end

%% parameters
Ft = 200e9;   %#ok<NASGU> % 采样率仅用于可选的谱分析
Osamp_factor = 2;
NumSymbols = 2^18;
M = 4;

randStream = RandStream('mt19937ar', 'Seed', 529558);
prevStream = RandStream.setGlobalStream(randStream);

%% MQAM modulate
[xsym, xm] = PAMSource(M, NumSymbols);
xs = xm;

%% pulse shaping（使用 rcosdesign 代替已弃用的 fdesign.pulseshaping）
rolloff = 0.1;
spanSymbols = 64; % 与原 128 taps、2x 上采样近似匹配（64*2+1 ≈ 129 taps）
sqrt_ht = rcosdesign(rolloff, spanSymbols, Osamp_factor, 'sqrt');
sqrt_ht = sqrt_ht./max(sqrt_ht);
x_upsamp = upsample(xs,Osamp_factor);%upsampling
x_shape = conv(sqrt_ht,x_upsamp);
x_shape = x_shape ./ sqrt(mean(abs(x_shape).^2));

tapLength = 111;
lambda = 0.9999;
defaultPreamble = 10000;

results = repmat(struct('label','', 'ber', NaN, 'snrDb', NaN, ...
    'ser', NaN, 'numTraining', NaN, 'numSamples', NaN, 'dataPath', '', 'figure', []), numel(dataFiles), 1);

for n1 = 1:numel(dataFiles)
    dataPath = dataFiles{n1};
    if ~isfile(dataPath)
        warning('数据文件 %s 不存在，已跳过。', dataPath);
        continue;
    end

    dataStruct = load(dataPath);
    if isfield(dataStruct, 'ReData')
        ReData = dataStruct.ReData;
    else
        error('文件 %s 中未找到 ReData 变量。', dataPath);
    end

    % 保持与原始脚本一致的符号方向
    ReData = -double(ReData(:));

    %% synchronization
    th = 0.3;
    [TE, ~] = TEFEMMM2(ReData,1024,th);
    ysync = ReData(1024+20+TE:end);

    available = min(length(ysync), length(x_shape));
    ysync = ysync(1:available);

    if length(ysync) <= N
        warning('数据 %s 可用长度不足，无法完成滤波。', dataPath);
        continue;
    end

    %% Match filtering
    yt_filter = ysync(1+N/2:length(ysync)-N/2);

    %% RLS LE/NE
    effectiveLength = min(length(yt_filter), length(xs));
    xTx = xs(1:effectiveLength);
    xRx = yt_filter(1:effectiveLength);

    NumPreamble_TDE = min(defaultPreamble, floor((effectiveLength-1)/2));
    if NumPreamble_TDE < 10
        warning('数据 %s 的训练序列过短（%d），已跳过。', dataPath, NumPreamble_TDE);
        continue;
    end

    [hffe,ye] = FFE_2pscenter(xRx,xTx,NumPreamble_TDE,tapLength,lambda); %#ok<NASGU>

    %% Normalize
    ym = Normalizepam(ye,M);

    %% MQAM demodulation
    ysym = dePAMSource(M,ym);

    %% BER/SNR
    refSym = xsym(1:length(ysym));
    validRange = NumPreamble_TDE+1:length(ysym);

    [~, BER_sub1] = biterr(ysym(validRange), refSym(validRange), log2(M));
    [~, SER_sub1] = symerr(ysym(validRange), refSym(validRange));         
    [SNRdB_sub1, ~] = snr( xTx(validRange), ym(validRange) );

    ytemp = ym(validRange);
    idealLevels = unique(xs);

    fig = figure('Name', sprintf('%s Equalization', dataPath), 'NumberTitle', 'off');
    subplot(2,2,1);
    plot(real(ytemp),'.'); grid on;
    title('均衡后符号幅度'); xlabel('符号索引'); ylabel('幅度');
    yline(idealLevels,'--r');

    subplot(2,2,2);
    histogram(real(ytemp),100,'Normalization','probability'); grid on;
    title('幅度分布'); xlabel('幅度'); ylabel('概率');
    xline(idealLevels,'--r');

    subplot(2,2,3);
    plot(validRange, real(xTx(validRange))-real(ym(validRange)));
    grid on; title('均衡误差'); xlabel('符号索引'); ylabel('误差');

    subplot(2,2,4);
    text(0.1,0.8,sprintf('BER = %.3e',BER_sub1),'FontSize',12);
    text(0.1,0.6,sprintf('SER = %.3e',SER_sub1),'FontSize',12);
    text(0.1,0.4,sprintf('SNR = %.2f dB',SNRdB_sub1),'FontSize',12);
    axis off;

    [~, label, ~] = fileparts(dataPath);
    results(n1).label = label;
    results(n1).ber = BER_sub1;
    results(n1).snrDb = SNRdB_sub1;
    results(n1).ser = SER_sub1;
    results(n1).numTraining = NumPreamble_TDE;
    results(n1).numSamples = effectiveLength;
    results(n1).dataPath = dataPath;
    results(n1).figure = fig;

    fprintf('%s: BER=%.3e, SNR=%.2f dB, 样本=%d\n', label, BER_sub1, SNRdB_sub1, effectiveLength);
end

validMask = ~isnan([results.ber]);
if any(validMask)
    figure('Name','PAM4 BER Summary','NumberTitle','off');
    bar(categorical({results(validMask).label}), [results(validMask).ber]);
    ylabel('BER'); grid on;
    title('各数据集 BER 对比');
end

RandStream.setGlobalStream(prevStream);

if nargout == 0
    assignin('base','loadpam4_results',results);
end
end
