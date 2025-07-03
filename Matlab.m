% ========================================================================
% V2G OPTIMIZATION SIMULATOR (REINFORCEMENT LEARNING ONLY)
% MATLAB SINGLE-FILE VERSION - FINAL FIX
% Author: Angelo Caravella 
%
% Description:
% This script simulates a Reinforcement Learning (RL) strategy for
% optimizing the charging and discharging of an electric vehicle (V2G).
% It includes battery degradation and user anxiety cost models.
% All necessary functions are included in this single file.
%
% To Run:
% 1. DELETE the 'q_tables' folder if it exists.
% 2. Place this file in a folder.
% 3. Create a subfolder named 'downloads'.
% 4. Place the 'PrezziZonali.xlsx' 
% 5. Run this script from MATLAB.
% ========================================================================

clear; clc; close all;

%% ========================================================================
% MAIN SCRIPT
% ========================================================================

% --- GLOBAL CONFIGURATION ---
VEHICLE_PARAMS = struct(...
    'capacita', 60, ...
    'p_carica', 7.4, ...
    'p_scarica', 5.0, ...
    'efficienza_carica', 0.95, ...
    'efficienza_scarica', 0.95, ...
    'soc_max', 0.9, ...
    'soc_min_batteria', 0.1, ...
    'degradation_model', 'nca', ...
    'costo_batteria', 150 * 60, ...
    'lfp_k_slope', 0.0035 ...
);

SIMULATION_PARAMS = struct(...
    'soc_min_utente', 0.3, ...
    'penalita_ansia', 0.01, ...
    'initial_soc', 0.5, ...
    'soc_target_finale', 0.5 ...
);

RL_PARAMS = struct(...
    'states_hour', 24, ...
    'states_soc', 11, ...
    'alpha', 0.1, ...
    'gamma', 0.98, ...
    'epsilon', 1.0, ...
    'epsilon_decay', 0.99985, ...
    'epsilon_min', 0.01, ...
    'episodes', 100000, ...
    'q_table_file', fullfile('q_tables', 'q_table_multiday_v1.mat') ...
);

% Combine all parameters into a single struct for easy passing to functions
ALL_PARAMS = struct('vehicle', VEHICLE_PARAMS, 'sim', SIMULATION_PARAMS, 'rl', RL_PARAMS);


% --- 1. LOAD AND PREPARE DATA ---
all_price_data = loadPriceData();
[training_profiles, test_profile] = splitData(all_price_data, 'Italia');


% --- 2. TRAIN OR LOAD RL AGENT ---
q_table_dir = fileparts(RL_PARAMS.q_table_file);
if ~isfolder(q_table_dir)
    mkdir(q_table_dir);
end

if isfile(RL_PARAMS.q_table_file)
    fprintf('\nLoading pre-trained Q-table from ''%s''...\n', RL_PARAMS.q_table_file);
    load(RL_PARAMS.q_table_file, 'q_table');
else
    fprintf('\nNo Q-table found. Starting RL training...\n');
    q_table = trainRLAgent(training_profiles, ALL_PARAMS);
    save(RL_PARAMS.q_table_file, 'q_table');
    fprintf('Q-table saved to ''%s''\n', RL_PARAMS.q_table_file);
end


% --- 3. EVALUATE THE RL STRATEGY ON THE TEST DAY ---
fprintf('\n%s\n', repmat('=', 1, 80));
fprintf('STARTING EVALUATION ON TEST DAY (Zone: Italia)\n');
fprintf('%s\n', repmat('=', 1, 80));

rl_results_table = runRLStrategy(test_profile, q_table, ALL_PARAMS);


% --- 4. SAVE RESULTS ---
output_dir = fileparts(RL_PARAMS.q_table_file);
if isempty(output_dir)
    output_dir = 'output';
end
saveResultsToExcel(rl_results_table, output_dir);

fprintf('\nSimulation finished.\n');


%% ========================================================================
% LOCAL HELPER FUNCTIONS
% ========================================================================

function df = loadPriceData(file_path)
    % Loads and cleans price data from the Excel file.
    if nargin < 1
        file_path = fullfile('PrezziZonali.xlsx');
    end
    
    fprintf('\n%s\n', repmat('=', 1, 80));
    fprintf('LOADING PRICE DATA\n');
    fprintf('%s\n', repmat('=', 1, 80));
    
    try
        opts = detectImportOptions(file_path, 'VariableNamingRule', 'preserve');
        df = readtable(file_path, opts);
        fprintf('File ''%s'' loaded successfully.\n', file_path);
    catch ME
        error('ERROR: File ''%s'' not found or cannot be read. Details: %s', file_path, ME.message);
    end
    
    if ~ismember('Ora', df.Properties.VariableNames)
        error('The Excel file must contain a column named ''Ora''.');
    end
    
    if iscell(df.Ora)
        df.Ora = str2double(df.Ora);
    end

    for col_idx = 1:width(df)
        col_name = df.Properties.VariableNames{col_idx};
        if ~ismember(col_name, {'Data', 'Ora'})
            try
                if iscell(df.(col_name))
                    df.(col_name) = str2double(strrep(df.(col_name), ',', '.'));
                end
                df.(col_name) = df.(col_name) / 1000;
            catch
                warning('Could not convert column ''%s''. It will be ignored.', col_name);
                df = removevars(df, col_name);
            end
        end
    end
end

function [training_profiles, test_profile] = splitData(df, test_zone)
    % Splits the data into training profiles and a single test profile.
    if ~ismember(test_zone, df.Properties.VariableNames)
        available_zones = strjoin(df.Properties.VariableNames(~ismember(df.Properties.VariableNames, {'Data', 'Ora'})), ', ');
        error('Test zone ''%s'' not found. Available zones: %s', test_zone, available_zones);
    end
    
    all_zones = df.Properties.VariableNames(~ismember(df.Properties.VariableNames, {'Data', 'Ora'}));
    training_zone_names = setdiff(all_zones, test_zone, 'stable');
    
    training_profiles = {};
    for i = 1:length(training_zone_names)
        zone = training_zone_names{i};
        profile = containers.Map('KeyType', 'double', 'ValueType', 'double');
        for r = 1:height(df)
            profile(df.Ora(r) - 1) = df.(zone)(r);
        end
        training_profiles{end+1} = profile;
    end
    
    test_profile = containers.Map('KeyType', 'double', 'ValueType', 'double');
    for r = 1:height(df)
        test_profile(df.Ora(r) - 1) = df.(test_zone)(r);
    end
    
    fprintf('\nData split: %d profiles for training, 1 profile (''%s'') for testing.\n', numel(training_profiles), test_zone);
end

function results = runRLStrategy(prices, q_table, params)
    % Manages the execution of the RL strategy simulation.
    fprintf('\nHourly Detail - Reinforcement Learning Strategy:\n');
    fprintf('%s | %s | %s | %s | %s | %s | %s | %s | %s | %s\n', ...
        'Hour', 'Initial SoC ', 'Price   ', 'Action   ', 'SoC Variation', 'Energy Cost', 'Energy Revenue', 'Degradation', 'Anxiety', 'Net Gain');
    fprintf('%s\n', repmat('-', 1, 150));

    actions_map = {'Wait', 'Charge', 'Discharge'};
    strategy_logic_fh = @(soc, hour) rl_logic(soc, hour, q_table, params, actions_map);

    results = runSimulationLoop(prices, params, strategy_logic_fh);
end

function [action_str, power_kwh] = rl_logic(soc, hour, q_table, p, actions_map)
    % Determines the action to take based on the Q-table.
    soc_states = p.rl.states_soc;
    soc_discrete = floor(soc * (soc_states - 1)) + 1;
    soc_discrete = max(1, min(soc_states, soc_discrete));
    
    q_values = squeeze(q_table(hour + 1, soc_discrete, :));
    
    max_q = max(q_values);
    best_action_indices = find(q_values == max_q);
    action_idx = best_action_indices(randi(length(best_action_indices)));
    
    action_str = actions_map{action_idx};
    power_kwh = 0;
    if strcmp(action_str, 'Charge')
        power_kwh = p.vehicle.p_carica;
    elseif strcmp(action_str, 'Discharge')
        power_kwh = p.vehicle.p_scarica;
    end
end

function results_table = runSimulationLoop(prices, params, strategy_logic_fh)
    % The core simulation engine that iterates through the day.
    soc = params.sim.initial_soc;
    log = {}; % Initialize log as an empty cell array
    
    cumulative_costs = struct('energy', 0, 'revenue', 0, 'degradation', 0, 'anxiety', 0);
    
    for hour = 0:23
        price = prices(hour);
        soc_start = soc;
        
        [action, power_kwh] = strategy_logic_fh(soc, hour);
        
        energy_cost = 0;
        energy_revenue = 0;
        energy_processed_kwh = 0;
        soc_end = soc_start;
        
        if strcmp(action, 'Charge')
            energy_stored_kwh = power_kwh * params.vehicle.efficienza_carica;
            soc_end = soc_start + energy_stored_kwh / params.vehicle.capacita;
            energy_cost = price * power_kwh;
            energy_processed_kwh = energy_stored_kwh;
        elseif strcmp(action, 'Discharge')
            energy_drawn_kwh = power_kwh / params.vehicle.efficienza_scarica;
            soc_end = soc_start - energy_drawn_kwh / params.vehicle.capacita;
            energy_revenue = price * power_kwh;
            energy_processed_kwh = -energy_drawn_kwh;
        end
        
        degradation_cost = calculateDegradationCost(soc_start, soc_end, energy_processed_kwh, params.vehicle);
        anxiety_cost = calculateAnxietyCost(soc_end, params.sim);
        net_gain = energy_revenue - energy_cost - degradation_cost - anxiety_cost;
        
        cumulative_costs.energy = cumulative_costs.energy + energy_cost;
        cumulative_costs.revenue = cumulative_costs.revenue + energy_revenue;
        cumulative_costs.degradation = cumulative_costs.degradation + degradation_cost;
        cumulative_costs.anxiety = cumulative_costs.anxiety + anxiety_cost;
        
        soc = max(0, min(1, soc_end));
        
        variation_soc = (soc - soc_start) * 100;
        fprintf('%4d | %12.1f%% | %8.4f | %-9s | %15.1f | %13.4f | %16.4f | %14.4f | %10.4f | %13.4f\n', ...
            hour + 1, soc_start*100, price, action, variation_soc, energy_cost, energy_revenue, ...
            degradation_cost, anxiety_cost, net_gain);
        
        % --- FIX: Construct the log row explicitly as a cell array ---
        log_row = {hour + 1, soc_start*100, price, {action}, variation_soc, ...
                   energy_cost, energy_revenue, degradation_cost, anxiety_cost, net_gain};
        log = [log; log_row]; % Append the new row
        % --- END FIX ---
    end
    
    final_soc_cost = calculateTerminalSOCCost(soc, params.sim);
    cumulative_costs.anxiety = cumulative_costs.anxiety + final_soc_cost;
    total_gain = cumulative_costs.revenue - cumulative_costs.energy - cumulative_costs.degradation - cumulative_costs.anxiety;
    
    fprintf('\nDAILY SUMMARY:\n');
    fprintf('  - Total Energy Cost: %.4f €\n', cumulative_costs.energy);
    fprintf('  - Total Energy Revenue: %.4f €\n', cumulative_costs.revenue);
    fprintf('  - Total Degradation Cost: %.4f €\n', cumulative_costs.degradation);
    fprintf('  - Total Anxiety Cost: %.4f €\n', cumulative_costs.anxiety);
    fprintf('  - TOTAL NET GAIN: %.4f €\n', total_gain);
    
    % This should now work without errors
    results_table = cell2table(log, 'VariableNames', {'Hour', 'Initial_SoC_perc', 'Price_EUR_kWh', 'Action', ...
        'SoC_Variation_perc', 'Energy_Cost_EUR', 'Energy_Revenue_EUR', 'Degradation_Cost_EUR', ...
        'Anxiety_Cost_EUR', 'Net_Gain_EUR'});
end

function q_table = trainRLAgent(training_profiles, params)
    % Trains the Q-learning agent.
    p_rl = params.rl;
    q_table = zeros(p_rl.states_hour, p_rl.states_soc, 3);
    epsilon = p_rl.epsilon;
    num_profiles = length(training_profiles);
    
    fprintf('Training RL agent on %d price profiles for %d episodes...\n', num_profiles, p_rl.episodes);
    
    for episode = 1:p_rl.episodes
        profile_idx = randi(num_profiles);
        episode_prices = training_profiles{profile_idx};
        soc = params.sim.initial_soc;
        
        for hour = 0:(p_rl.states_hour - 1)
            soc_discrete = floor(soc * (p_rl.states_soc - 1)) + 1;
            soc_discrete = max(1, min(p_rl.states_soc, soc_discrete));
            
            if rand() < epsilon
                action_idx = randi(3);
            else
                q_values = squeeze(q_table(hour + 1, soc_discrete, :));
                max_q = max(q_values);
                best_action_indices = find(q_values == max_q);
                action_idx = best_action_indices(randi(length(best_action_indices)));
            end
            
            [new_soc, reward] = getRLStepResult(soc, hour, action_idx, episode_prices, params);
            new_soc_discrete = floor(new_soc * (p_rl.states_soc - 1)) + 1;
            new_soc_discrete = max(1, min(p_rl.states_soc, new_soc_discrete));
            
            if hour < p_rl.states_hour - 1
                next_max_q = max(q_table(hour + 2, new_soc_discrete, :));
            else
                next_max_q = 0;
            end
            
            current_q = q_table(hour + 1, soc_discrete, action_idx);
            new_q = (1 - p_rl.alpha) * current_q + p_rl.alpha * (reward + p_rl.gamma * next_max_q);
            q_table(hour + 1, soc_discrete, action_idx) = new_q;
            
            soc = new_soc;
        end
        
        epsilon = max(p_rl.epsilon_min, epsilon * p_rl.epsilon_decay);
        
        if mod(episode, 20000) == 0
            fprintf('Episode %d/%d, Epsilon: %.4f\n', episode, p_rl.episodes, epsilon);
        end
    end
    
    fprintf('RL Training complete!\n');
end

function [new_soc, reward] = getRLStepResult(soc, hour, action_idx, prices, params)
    % Calculates the result of a single action during training.
    p_vehicle = params.vehicle;
    p_sim = params.sim;
    
    new_soc = soc;
    reward = 0;
    energy_processed_kwh = 0;
    price = prices(hour);
    
    if action_idx == 2 && soc < p_vehicle.soc_max % Charge
        energy_stored = p_vehicle.p_carica * p_vehicle.efficienza_carica;
        new_soc = soc + energy_stored / p_vehicle.capacita;
        reward = reward - price * p_vehicle.p_carica;
        energy_processed_kwh = energy_stored;
    elseif action_idx == 3 && soc > p_vehicle.soc_min_batteria % Discharge
        energy_drawn = p_vehicle.p_scarica / p_vehicle.efficienza_scarica;
        new_soc = soc - energy_drawn / p_vehicle.capacita;
        reward = reward + price * p_vehicle.p_scarica;
        energy_processed_kwh = -energy_drawn;
    end
    
    new_soc = max(0, min(1, new_soc));
    
    reward = reward - calculateDegradationCost(soc, new_soc, energy_processed_kwh, p_vehicle);
    reward = reward - calculateAnxietyCost(new_soc, p_sim);
    
    is_last_hour = (hour == params.rl.states_hour - 1);
    if is_last_hour
        reward = reward - calculateTerminalSOCCost(new_soc, p_sim);
    end
end

function cost = calculateDegradationCost(soc_start, soc_end, energy_kwh, vehicle_params)
    % Calculates battery degradation cost based on the selected model.
    switch vehicle_params.degradation_model
        case 'nca'
            if soc_end >= soc_start
                cost = 0;
                return;
            end
            dod_start = 1.0 - soc_start;
            dod_end = 1.0 - soc_end;
            inv_phi_start = cycle_life_phi_nca(dod_start);
            inv_phi_end = cycle_life_phi_nca(dod_end);
            cost = (inv_phi_end - inv_phi_start) * vehicle_params.costo_batteria;
        case 'lfp'
            k = vehicle_params.lfp_k_slope;
            cost = (abs(energy_kwh) / vehicle_params.capacita) * (k / 100) * vehicle_params.costo_batteria;
        otherwise % 'simple'
            cost = 0.008 * abs(energy_kwh);
    end
end

function phi = cycle_life_phi_nca(dod)
    % Helper for the NCA degradation model.
    dod_perc = dod * 100;
    if dod_perc <= 0
        phi = 0;
    else
        phi = 6.6e-6 * exp(0.045 * dod_perc);
    end
end

function cost = calculateAnxietyCost(soc, sim_params)
    % Calculates the cost associated with user's range anxiety.
    cost = 0;
    if soc < sim_params.soc_min_utente
        cost = sim_params.penalita_ansia * (sim_params.soc_min_utente - soc) * 100;
    end
end

function cost = calculateTerminalSOCCost(soc, sim_params)
    % Calculates the penalty for not meeting the final SoC target.
    cost = 0;
    target = sim_params.soc_target_finale;
    if soc < target
        cost = sim_params.penalita_ansia * 5.0 * (target - soc) * 100;
    end
end

function saveResultsToExcel(results_df, output_dir)
    % Saves the final results table to an Excel file.
    if ~isfolder(output_dir)
        mkdir(output_dir);
    end
    output_path = fullfile(output_dir, 'Results_V2G_RL_Only_MATLAB.xlsx');
    
    try
        writetable(results_df, output_path, 'Sheet', 'Reinforcement Learning', 'WriteRowNames', false);
        fprintf('\nDetailed results saved to: %s\n', output_path);
    catch ME
        fprintf('\nERROR: Could not save results to Excel file. Please check permissions.\n');
        fprintf('Details: %s\n', ME.message);
    end
end