%% PAM4 main
clear;
close all;

%% add paths
addpath('fns');
addpath(fullfile('fns','fns2'));

%% parameters
Ft = 200e9;
Osamp_factor = 2;
NumSymbols = 2^18;
NumPreamble = 0;
NumSym_total = NumSymbols + NumPreamble;
M = 4;

s = RandStream.create('mt19937ar', 'seed', 529558);
prevStream = RandStream.setGlobalStream(s);

%% PAM4 modulate
[xsym, xm] = PAMSource(M, NumSymbols);
xsym = xsym(:);
xm = xm(:);
xs = xm;

%% pulse shaping
rolloff = 0.1;
N = 128;
h = fdesign.pulseshaping(Osamp_factor, 'Raised Cosine', 'N,Beta', N, rolloff);
Hd = design(h);
sqrt_ht = Hd.Numerator;
sqrt_ht = sqrt_ht ./ max(sqrt_ht);

x_upsamp = upsample(xs, Osamp_factor);
x_shape = conv(sqrt_ht, x_upsamp);
x_shape = x_shape ./ sqrt(mean(abs(x_shape).^2));

%% data list
NN = 101;
file_list = {'rop3dBm_1.mat', 'rop5dBm_1.mat'};
BERall = zeros(length(file_list), length(NN));

for n1 = 1:length(file_list)
    disp(['Processing File Index: ', num2str(n1)]);
    load(file_list{n1}, 'ReData')

    ReData = -ReData;

    %% synchronization
    th = 0.3;
    [TE, FE] = TEFEMMM2(ReData, 1024, th);
    abc = TE + 0;
    TE = abc;

    ysync = ReData(1024 + 20 + TE : end);
    ysync = ysync(1 : length(x_shape));

    for m1 = 1:length(NN)
        %% match filtering
        yt_filter = ysync(1 + N/2 : length(ysync) - N/2);

        %% equalizer inputs
        xTx = xs;
        xRx = yt_filter;
        NumPreamble_TDE = 10000;

        %% equalizer parameters
        N1 = 111;
        N2 = 21;
        WL = 1;
        D1 = 25;
        D2 = 0;
        WD = 1;

        K_Lin = 18;
        K_Vol = 90;

        aa = 0.3;
        bb = 1;

        %% algorithm switching (uncomment one)
        clear ye ye_valid valid_idx net out stats d0 idxTx ye_use
        %algo_id = 'FFE'; % set manually to match the algorithm you run

       %  [hffe, ye] = FFE_2pscenter(xRx, xTx, NumPreamble_TDE, N1, 0.9999);
        % algo_id = 'FFE';

        % [hffe, ye] = VNLE2_2pscenter(xRx, xTx, NumPreamble_TDE, N1, N2, 0.9999, WL);
        % algo_id = 'VNLE';

        % [hffe, hdfe, ye] = LE_FFE2ps_centerDFE_new(xRx, xTx, NumPreamble_TDE, N1, D1, 0.9999, M, M/2);
        % algo_id = 'LE_FFE_DFE';

        % [hffe, hdfe, ye] = DP_VFFE2pscenter_VDFE(xRx, xTx, NumPreamble_TDE, N1, N2, D1, D2, 0.9999, WL, WD, M, M/2);
        % algo_id = 'DP_VFFE_VDFE';

        % [BER_clut, ye] = CLUT_VDFE_Implementation(xRx, xTx, NumPreamble_TDE, N1, N2, D1, D2, WL, WD, M, K_Lin, K_Vol, 0.9999);
        % algo_id = 'CLUT_VDFE';

         [ye_valid, net, valid_idx] = FNN_Implementation(xRx, xTx, NumPreamble_TDE, 101, 64, 0.001, 30);
         algo_id = 'FNN';

         %[ye, net, valid_idx] = RNN_Implementation(xRx, xTx, NumPreamble_TDE, 41, 64, 0.001, 15, 2, [8 10], [1 2]);
         %algo_id = 'RNN';

        %% unified output mapping
        if isempty(algo_id)
            if exist('ye_valid', 'var') || exist('valid_idx', 'var')
                algo_id = 'FNN'; % assume NN-type if valid_idx/ye_valid exist
            elseif exist('ye', 'var')
                algo_id = 'FFE'; % default to classical if only ye exists
            else
                error('algo_id is empty and no equalizer output found.');
            end
        end

        switch upper(algo_id)
            case {'FNN','RNN'}
                if ~exist('valid_idx', 'var') || isempty(valid_idx)
                    error('valid_idx missing for NN-type algorithm.');
                end
                idxTx = valid_idx(:);
                if exist('ye_valid', 'var') && ~isempty(ye_valid)
                    ye_use = ye_valid(:);
                else
                    ye_use = ye(idxTx);
                end
            otherwise
                if ~exist('ye', 'var') || isempty(ye)
                    error('ye missing for classical algorithm. Uncomment the algorithm call.');
                end
                [off, d0] = align_offset_delay_by_ser(ye, xsym, NumPreamble_TDE, M, -60:60);
                if length(ye) > 1.5 * length(xsym)
                    ye_use = ye(off:2:end);
                else
                    ye_use = ye(:);
                end
                idxTx = (1:length(ye_use)).' + d0;
        end

        out.ye = ye_use;
        out.idxTx = idxTx;
        out.name = algo_id;

        %% BER/SNR (unified)
        stats = eval_equalizer_pam4(out.ye, out.idxTx, xsym, xm, NumPreamble_TDE, M);
        BER_sub1 = stats.BER;
        SNRdB_sub1 = stats.SNRdB;

        disp(['File ', num2str(n1), ' BER = ', num2str(BER_sub1)]);
        disp(['File ', num2str(n1), ' SNR (dB) = ', num2str(SNRdB_sub1)]);

        BERall(n1, m1) = BER_sub1;
    end
end

mean_ber = mean(BERall, 1);
disp(['Average BER = ', num2str(mean_ber)]);
return;

if 0
    [aaa1, bbb1] = sort(BERall(:,end));
    avernber = mean(BERall(bbb1(1:10),:), 1);
end

berm = mean(BERall, 1);
figure;semilogy(NN, berm);
