%% Replot from Existing FIG file
clear; close all;

fig_filename = '2km.fig';
if ~exist(fig_filename, 'file')
    error('File %s not found!', fig_filename);
end

% Open the figure invisibly first to extract data
f_old = openfig(fig_filename, 'invisible');
ax_old = gca(f_old);

% Get all line objects
lines = findobj(ax_old, 'Type', 'line');
% Get legend
leg = findobj(f_old, 'Type', 'legend');

if isempty(leg)
    warning('No legend found in fig. Data matching might be hard.');
    algo_names = {};
else
    algo_names = leg.String;
end

% Extract Data
% Note: 'lines' are usually in reverse plotting order (stack order)
% We need to map lines to legend entries.
% Usually Legend order matches Child order or reverse.

% Let's store data in a struct
extracted_data = struct();
num_lines = length(lines);

% Reverse lines to match plotting order (usually)
% lines = flipud(lines); 

disp(['Found ', num2str(num_lines), ' lines in figure.']);

% Prepare container
% We expect 8 lines for algorithms + maybe threshold lines
real_lines = [];
for i = 1:num_lines
    % Filter out threshold lines (usually no markers or specific color)
    if ~isempty(lines(i).Marker) && ~strcmp(lines(i).Marker, 'none')
        real_lines = [real_lines; lines(i)];
    end
end

% Sort real lines by Y-value at last point (high BER to low BER) to guess Algo
% FFE is usually high, WDRNN is low.
% But better to trust the legend if possible.

% If legend exists, let's assume 1:1 mapping with plotted lines (excluding threshold lines)
% Standard MATLAB Legend associates with 'Children' of Axes.
% Let's try to grab display names
for i = 1:length(real_lines)
    x = real_lines(i).XData;
    y = real_lines(i).YData;
    
    % Check if Y is log10 (negative values) or raw
    if any(y < 0)
        % It's likely log10(BER)
        y = 10.^y;
    end
    
    % Store
    extracted_data(i).x = x;
    extracted_data(i).y = y;
    extracted_data(i).name = real_lines(i).DisplayName;
    
    if isempty(extracted_data(i).name) && i <= length(algo_names)
        % Fallback to legend string if DisplayName is empty
        % Note: Legend usually lists top-to-bottom, which corresponds to 
        % plotted lines in specific order. 
        % Often lines(end) is the first plotted (FFE).
        % Let's print mean BER to verify manually
    end
end

close(f_old);

%% Plot New Figure
figure('Name', 'Re-plotted BER (Semilogy)', 'Color', 'w', 'Position', [100, 100, 800, 600]);

% Standard Algo List Order (for coloring)
std_algo_list = {'FFE', 'VNLE', 'LE_FFE_DFE', 'DP_VFFE_VDFE', 'CLUT_VDFE', 'FNN', 'RNN', 'WDRNN'};
colors = [
    0 0.4470 0.7410;      % Blue
    0.8500 0.3250 0.0980; % Red
    0.9290 0.6940 0.1250; % Yellow
    0.4940 0.1840 0.5560; % Purple
    0.4660 0.6740 0.1880; % Green
    0.3010 0.7450 0.9330; % Cyan
    0.5 0.5 0.5;          % Gray
    0 0 0                 % Black
];
markers = {'o-', 's-', 'd-', '^-', 'v-', '>-', 'p-', 'h-'};

hold on;

% We need to match extracted lines to algorithms
% Heuristic: Match based on mean BER value?
% Or simply plot what we found and let user check legend.
% Let's try to match by DisplayName if available.

for i = 1:length(extracted_data)
    x = extracted_data(i).x;
    y = extracted_data(i).y;
    name = extracted_data(i).name;
    
    % Heuristic Identification if name is missing
    if isempty(name)
        % Try to identify based on y-value at 5.6dBm (last point)
        last_y = y(end);
        if last_y > 5e-2, name_guess = 'FFE/VNLE/FNN?';
        elseif last_y > 4e-3, name_guess = 'DFE?';
        elseif last_y > 2.5e-3, name_guess = 'VDFE/RNN?';
        elseif last_y < 2.5e-3, name_guess = 'WDRNN!'; 
        else name_guess = ['Unknown-' num2str(i)];
        end
        disp(['Line ', num2str(i), ' (Last Val=', num2str(last_y, '%.2e'), ') -> Guess: ', name_guess]);
        
        % Assign color based on value rank?
        % Let's just plot with default cycle if unknown
        style_idx = mod(i-1, length(colors)) + 1;
        col = colors(style_idx, :);
        mk = markers{style_idx};
    else
        % Match name to std_algo_list to get consistent color
        idx = find(strcmpi(std_algo_list, name));
        if isempty(idx)
             % Try partial match
             idx = find(contains(std_algo_list, name, 'IgnoreCase', true));
        end
        
        if ~isempty(idx)
            col = colors(idx(1), :);
            mk = markers{idx(1)};
        else
            col = [0 0 0]; mk = 'x-'; % Unknown
        end
    end
    
    % If identification failed, use loop index style (reverse because lines are usually reverse)
    if isempty(name)
        % lines are usually LIFO (Last plotted is First in array)
        % So Line 1 in array is likely the Last Algorithm (WDRNN)
        % Line 8 is FFE.
        % Let's verify: Line 1 (i=1) usually has lowest BER?
        % Let's invert index for style
        style_idx = length(std_algo_list) - i + 1;
        if style_idx < 1, style_idx = 1; end
        col = colors(style_idx, :);
        mk = markers{style_idx};
        
        % Assign Name based on order
        final_name = std_algo_list{style_idx};
    else
        final_name = name;
    end

    semilogy(x, y, mk, 'Color', col, 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', col, 'DisplayName', final_name);
end

% Thresholds
yline(3.8e-3, '--k', 'HD-FEC (3.8e-3)', 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'left');
yline(2.4e-2, ':k', 'SD-FEC (2.4e-2)', 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'left');

grid on; grid minor;
xlabel('Received Optical Power (dBm)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('BER', 'FontSize', 12, 'FontWeight', 'bold');
title('BER Performance vs ROP (200Gbps 2km)', 'FontSize', 14);
legend('show', 'Location', 'southwest');
ylim([1e-3, 2e-1]);

disp('Replot complete. Please verify the legend mapping.');
