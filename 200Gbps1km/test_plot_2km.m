%% Test Plot 2km Data (Better Visualization)
clear; close all;

% Load Data
if ~exist('BER_Results_2km.mat', 'file')
    error('BER_Results_2km.mat not found! Did you run roll_8_2km?');
end
load('BER_Results_2km.mat');

% Prepare Figure
figure('Name', 'BER vs ROP (2km) - Semilogy', 'Color', 'w', 'Position', [100, 100, 800, 600]);

% Color Palette & Markers (Updated for Clarity)
colors = [
    0 0.4470 0.7410;      % 1:FFE (Blue)
    0.8500 0.3250 0.0980; % 2:VNLE (Red)
    0.9290 0.6940 0.1250; % 3:DFE (Yellow)
    0.4940 0.1840 0.5560; % 4:VDFE (Purple)
    0.4660 0.6740 0.1880; % 5:CLUT (Green)
    0.3010 0.7450 0.9330; % 6:FNN (Cyan)
    0.5 0.5 0.5;          % 7:RNN (Gray)
    0 0 0                 % 8:WDRNN (Black)
];
markers = {'o-', 's-', 'd-', '^-', 'v-', '>-', 'p-', 'h-'};

% Plot Loop
hold on;
h_plots = gobjects(length(algo_list), 1);

for a = 1:length(algo_list)
    y_data = BERall(:, a);
    % Avoid 0 for log plot
    y_data(y_data <= 0) = 1e-7; 
    
    % Use semilogy directly
    h_plots(a) = semilogy(rop_dBm, y_data, markers{a}, ...
        'Color', colors(a,:), ...
        'LineWidth', 2, ...         % Thicker lines
        'MarkerSize', 8, ...        % Bigger markers
        'MarkerFaceColor', colors(a,:));
end

% Threshold Lines
yline(3.8e-3, '--k', 'HD-FEC (3.8e-3)', 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'left', 'FontSize', 11);
yline(2.4e-2, ':k', 'SD-FEC (2.4e-2)', 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'left', 'FontSize', 11);

% Aesthetics
grid on;
grid minor; % Minor grid helps separate points visually
xlabel('Received Optical Power (dBm)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('BER (Log Scale)', 'FontSize', 14, 'FontWeight', 'bold');
title('BER Performance vs ROP (200Gbps 2km)', 'FontSize', 16);

% Set Limits to avoid "sticking"
xlim([1.4, 5.8]); % Add some padding to x-axis
ylim([1e-3, 2e-1]); % Focus on the relevant range (1e-1 to 1e-3)
% If WDRNN goes lower, adjust ylim, e.g., [1e-4, 2e-1]

% Legend
legend(h_plots, algo_list, 'Location', 'southwest', 'FontSize', 11, 'Interpreter', 'none');

set(gca, 'FontSize', 12, 'LineWidth', 1.2);
box on;
hold off;

disp('Plot generated. Please check the figure window.');
