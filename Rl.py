# ========================================================================
# V2G OPTIMIZATION SIMULATOR (REINFORCEMENT LEARNING ONLY)
# Author: Angelo Caravella
# Version: Academic v2.4 (RL-focused, English UI)
# Description: This script simulates a Reinforcement Learning strategy for
#              optimizing the charging and discharging of an electric
#              vehicle (Vehicle-to-Grid, V2G), incorporating battery
#              degradation models and user preferences.
#
# Key Academic References:
# 1. Ortega-Vazquez, M. A. (2014). "Optimal scheduling of electric
#    vehicle charging and vehicle-to-grid services at household level..."
# 3. Abdullah, H. M., et al. (2021). "Reinforcement Learning Based EV
#    Charging Management Systems-A Review".
# 4. Lee, S. T., et al. (2018). "A User-centric Control Scheme for
#    Residential V2G Operation to Mitigate Range Anxiety".
# ========================================================================

import numpy as np
import pandas as pd
import random
import os
import sys
from typing import Dict, List, Tuple

# ========================================================================
# GLOBAL CONFIGURATION
# ========================================================================

VEHICLE_PARAMS = {
    'capacita': 60,
    'p_carica': 7.4,
    'p_scarica': 5.0,
    'efficienza_carica': 0.95,
    'efficienza_scarica': 0.95,
    'soc_max': 0.9,
    'soc_min_batteria': 0.1,
    'degradation_model': 'nca',
    'costo_batteria': 150 * 60,
    'lfp_k_slope': 0.0035,
}

SIMULATION_PARAMS = {
    'soc_min_utente': 0.3,
    'penalita_ansia': 0.01,
    'initial_soc': 0.5,
    'soc_target_finale': 0.5,
}

RL_PARAMS = {
    'states_hour': 24,
    'states_soc': 11,
    'alpha': 0.1,
    'gamma': 0.98,
    'epsilon': 1.0,
    'epsilon_decay': 0.99985,
    'epsilon_min': 0.01,
    'episodes': 100000,
    'q_table_file': os.path.join('q_tables', 'q_table_multiday_v1.npy')
}

# ========================================================================
# BATTERY DEGRADATION MODEL CLASS
# ========================================================================

class BatteryDegradationModel:
    def __init__(self, vehicle_config: Dict):
        self.vehicle_params = vehicle_config
        self.battery_cost = vehicle_config['costo_batteria']
        self.battery_capacity = vehicle_config['capacita']

    def _cycle_life_phi_nca(self, dod: float) -> float:
        dod_perc = dod * 100
        if dod_perc <= 0: return 0.0
        return 6.6e-6 * np.exp(0.045 * dod_perc)

    def cost_simple_linear(self, energy_kwh: float) -> float:
        return 0.008 * abs(energy_kwh)

    def cost_lfp_model(self, energy_kwh: float) -> float:
        k = self.vehicle_params['lfp_k_slope']
        return (abs(energy_kwh) / self.battery_capacity) * (k / 100) * self.battery_cost

    def cost_nca_model(self, soc_start: float, soc_end: float) -> float:
        if soc_end >= soc_start: return 0.0
        dod_start = 1.0 - soc_start
        dod_end = 1.0 - soc_end
        inv_phi_start = self._cycle_life_phi_nca(dod_start)
        inv_phi_end = self._cycle_life_phi_nca(dod_end)
        return (inv_phi_end - inv_phi_start) * self.battery_cost

# ========================================================================
# V2G OPTIMIZER CLASS
# ========================================================================

class V2GOptimizer:
    def __init__(self, vehicle_config: Dict, sim_config: Dict):
        self.vehicle_params = vehicle_config
        self.sim_params = sim_config
        
        self.actions_map = {0: 'Wait', 1: 'Charge', 2: 'Discharge'}
        self.degradation_calculator = BatteryDegradationModel(self.vehicle_params)
        self.degradation_model_type = self.vehicle_params.get('degradation_model', 'simple')
        self.anxiety_penalty_per_perc = self.sim_params['penalita_ansia']
        self.soc_min_utente = self.sim_params['soc_min_utente']
        self.soc_target_finale = self.sim_params['soc_target_finale']
        self.current_prices = None

    def set_prices_for_simulation(self, prices: Dict[int, float]):
        self.current_prices = prices

    def _calculate_degradation_cost(self, soc_start: float, soc_end: float, energy_kwh: float) -> float:
        if self.degradation_model_type == 'lfp':
            return self.degradation_calculator.cost_lfp_model(energy_kwh)
        elif self.degradation_model_type == 'nca':
            return self.degradation_calculator.cost_nca_model(soc_start, soc_end)
        else:
            return self.degradation_calculator.cost_simple_linear(energy_kwh)

    def _calculate_anxiety_cost(self, soc: float) -> float:
        if soc < self.soc_min_utente:
            return self.anxiety_penalty_per_perc * (self.soc_min_utente - soc) * 100
        return 0.0

    def _calculate_terminal_soc_cost(self, soc: float) -> float:
        target = self.soc_target_finale
        if soc < target:
            return self.anxiety_penalty_per_perc * 5.0 * (target - soc) * 100
        return 0.0

    def _log_hourly_data(self, log: List, hour: int, soc_start: float, soc_end: float, price: float, action: str,
                         energy_cost: float, energy_revenue: float, degradation_cost: float,
                         anxiety_cost: float, net_gain: float):
        variation_soc = (soc_end - soc_start) * 100
        log.append({
            'Hour': hour + 1,
            'Initial SoC (%)': round(soc_start * 100, 1),
            'Price (€/kWh)': round(price, 4),
            'Action': action,
            'SoC Variation (%)': round(variation_soc, 1),
            'Energy Cost (€)': round(energy_cost, 4),
            'Energy Revenue (€)': round(energy_revenue, 4),
            'Degradation Cost (€)': round(degradation_cost, 4),
            'Anxiety Cost (€)': round(anxiety_cost, 4),
            'Net Gain (€)': round(net_gain, 4)
        })
        # This print format is designed to match the provided image's alignment
        print(f"{hour+1:3d} | {soc_start*100:11.1f}% | {price:8.4f} | {action:8} | "
              f"{variation_soc:13.1f} | {energy_cost:13.4f} | {energy_revenue:16.4f} | "
              f"{degradation_cost:14.4f} | {anxiety_cost:7.4f} | {net_gain:14.4f}")

    def _run_simulation_loop(self, strategy_logic):
        soc = self.sim_params['initial_soc']
        log = []
        cumulative_costs = {'energy': 0, 'revenue': 0, 'degradation': 0, 'anxiety': 0}

        for hour, price in self.current_prices.items():
            if hour >= 24: continue

            soc_start = soc
            action, power_kwh = strategy_logic(soc, hour)
            
            energy_cost, energy_revenue, degradation_cost = 0, 0, 0
            soc_end = soc_start
            energy_processed_kwh = 0

            if action == 'Charge':
                energy_stored_kwh = power_kwh * self.vehicle_params['efficienza_carica']
                soc_end += energy_stored_kwh / self.vehicle_params['capacita']
                energy_cost = price * power_kwh
                energy_processed_kwh = energy_stored_kwh
            elif action == 'Discharge':
                energy_drawn_kwh = power_kwh / self.vehicle_params['efficienza_scarica']
                soc_end -= energy_drawn_kwh / self.vehicle_params['capacita']
                energy_revenue = price * power_kwh
                energy_processed_kwh = -energy_drawn_kwh

            degradation_cost = self._calculate_degradation_cost(soc_start, soc_end, energy_processed_kwh)
            anxiety_cost = self._calculate_anxiety_cost(soc_end)
            net_gain = energy_revenue - energy_cost - degradation_cost - anxiety_cost
            
            cumulative_costs['energy'] += energy_cost
            cumulative_costs['revenue'] += energy_revenue
            cumulative_costs['degradation'] += degradation_cost
            cumulative_costs['anxiety'] += anxiety_cost
            
            soc = np.clip(soc_end, 0, 1)
            
            self._log_hourly_data(log, hour, soc_start, soc, price, action, energy_cost, energy_revenue,
                                  degradation_cost, anxiety_cost, net_gain)
        
        final_soc_cost_summary = self._calculate_terminal_soc_cost(soc)
        cumulative_costs['anxiety'] += final_soc_cost_summary
        
        return self._finalize_log(log, cumulative_costs)

    def train_rl_agent(self, rl_params: Dict, training_price_profiles: List[Dict[int, float]]) -> np.ndarray:
        print(f"\nTraining RL agent on {len(training_price_profiles)} price profiles ({rl_params['episodes']} episodes)...")
        q_table = np.zeros((rl_params['states_hour'], rl_params['states_soc'], len(self.actions_map)))
        epsilon = rl_params['epsilon']
        
        for episode in range(rl_params['episodes']):
            episode_prices = random.choice(training_price_profiles)
            soc = self.sim_params['initial_soc']
            for hour in range(rl_params['states_hour']):
                soc_discrete = self._discretize_soc(soc, rl_params['states_soc'])
                
                if random.random() < epsilon:
                    action = random.randint(0, len(self.actions_map) - 1)
                else:
                    action = np.argmax(q_table[hour, soc_discrete])
                
                new_soc, reward = self._get_rl_step_result(soc, hour, action, episode_prices, last_hour=(hour == rl_params['states_hour'] - 1))
                new_soc_discrete = self._discretize_soc(new_soc, rl_params['states_soc'])
                
                next_q_value = 0
                if hour < rl_params['states_hour'] - 1:
                    next_q_value = np.max(q_table[hour + 1, new_soc_discrete])
                
                current_q = q_table[hour, soc_discrete, action]
                new_q = (1 - rl_params['alpha']) * current_q + rl_params['alpha'] * (reward + rl_params['gamma'] * next_q_value)
                q_table[hour, soc_discrete, action] = new_q
                soc = new_soc
            
            epsilon = max(rl_params['epsilon_min'], epsilon * rl_params['epsilon_decay'])
            if (episode + 1) % 20000 == 0:
                print(f"Episode {episode + 1}/{rl_params['episodes']}, Epsilon: {epsilon:.4f}")
                
        print("RL Training complete!")
        q_table_dir = os.path.dirname(rl_params['q_table_file'])
        if q_table_dir and not os.path.exists(q_table_dir):
            os.makedirs(q_table_dir)
        np.save(rl_params['q_table_file'], q_table)
        print(f"Q-table saved to '{rl_params['q_table_file']}'")
        return q_table

    def _get_rl_step_result(self, soc: float, hour: int, action: int, prices: Dict[int, float], last_hour: bool) -> Tuple[float, float]:
        new_soc = soc
        reward = 0.0
        energy_processed_kwh = 0
        
        if action == 1 and soc < self.vehicle_params['soc_max']: # Charge
            energy_stored = self.vehicle_params['p_carica'] * self.vehicle_params['efficienza_carica']
            new_soc += energy_stored / self.vehicle_params['capacita']
            reward -= prices.get(hour, 0) * self.vehicle_params['p_carica']
            energy_processed_kwh = energy_stored
        elif action == 2 and soc > self.vehicle_params['soc_min_batteria']: # Discharge
            energy_drawn = self.vehicle_params['p_scarica'] / self.vehicle_params['efficienza_scarica']
            new_soc -= energy_drawn / self.vehicle_params['capacita']
            reward += prices.get(hour, 0) * self.vehicle_params['p_scarica']
            energy_processed_kwh = -energy_drawn
        
        reward -= self._calculate_degradation_cost(soc, new_soc, energy_processed_kwh)
        reward -= self._calculate_anxiety_cost(new_soc)
        if last_hour:
            reward -= self._calculate_terminal_soc_cost(new_soc)
        
        return np.clip(new_soc, 0, 1), reward

    def run_rl_strategy(self, q_table: np.ndarray) -> pd.DataFrame:
        print("\nHourly Detail - Reinforcement Learning Strategy:")
        # Print header to match the image format
        print(f"{'Hour':>3s} | {'Initial SoC':>11s} | {'Price':>8s} | {'Action':>8s} | "
              f"{'SoC Variation':>13s} | {'Energy Cost':>13s} | {'Energy Revenue':>16s} | "
              f"{'Degradation':>14s} | {'Anxiety':>7s} | {'Net Gain':>14s}")
        print(f"{'-'*3:s}-+-{'-'*11:s}-+-{'-'*8:s}-+-{'-'*8:s}-+-"
              f"{'-'*13:s}-+-{'-'*13:s}-+-{'-'*16:s}-+-"
              f"{'-'*14:s}-+-{'-'*7:s}-+-{'-'*14:s}")
        
        def logic(soc, hour):
            soc_discrete = self._discretize_soc(soc, q_table.shape[1])
            # Choose the best action, breaking ties randomly
            best_actions = np.where(q_table[hour, soc_discrete] == np.max(q_table[hour, soc_discrete]))[0]
            action_idx = np.random.choice(best_actions)
            
            action_str = self.actions_map[action_idx]
            power_kwh = 0
            if action_str == 'Charge':
                power_kwh = self.vehicle_params['p_carica']
            elif action_str == 'Discharge':
                power_kwh = self.vehicle_params['p_scarica']
            
            return action_str, power_kwh

        return self._run_simulation_loop(logic)

    def _discretize_soc(self, soc: float, states: int) -> int:
        return int(np.clip(round(soc * (states - 1)), 0, states - 1))

    def _finalize_log(self, log: List, cumulative_costs: Dict) -> pd.DataFrame:
        total_gain = cumulative_costs['revenue'] - cumulative_costs['energy'] - cumulative_costs['degradation'] - cumulative_costs['anxiety']
        print("\nDAILY SUMMARY:")
        print(f"  - Total Energy Cost: {cumulative_costs['energy']:.4f} €")
        print(f"  - Total Energy Revenue: {cumulative_costs['revenue']:.4f} €")
        print(f"  - Total Degradation Cost: {cumulative_costs['degradation']:.4f} €")
        print(f"  - Total Anxiety Cost: {cumulative_costs['anxiety']:.4f} €")
        print(f"  - TOTAL NET GAIN: {total_gain:.4f} €")
        
        if not log: return pd.DataFrame()
        
        return pd.DataFrame(log)

# ========================================================================
# HELPER FUNCTIONS
# ========================================================================

def load_price_data(file_path: str = None) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("LOADING PRICE DATA")
    print("=" * 80)
    
    script_dir = os.path.dirname(__file__) if '__file__' in locals() else os.getcwd()
    default_path = os.path.join(script_dir, "downloads", "PrezziZonali.xlsx")
    file_path = file_path if file_path else default_path
    
    try:
        df = pd.read_excel(file_path)
        print(f"File '{file_path}' loaded successfully.")
        
        if 'Ora' not in df.columns:
            raise ValueError("The Excel file must contain an 'Ora' column.")
        
        for col in df.columns:
            if col not in ['Ora', 'Data']:
                try:
                    if df[col].dtype == 'object':
                        df[col] = df[col].str.replace(',', '.', regex=False).astype(float)
                    df[col] = df[col] / 1000 # Convert from €/MWh to €/kWh
                except (ValueError, AttributeError) as e:
                    print(f"Warning: could not convert column '{col}'. It will be ignored. Error: {e}")
                    df.drop(columns=[col], inplace=True)
        
        return df

    except FileNotFoundError:
        print(f"ERROR: The file '{file_path}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Could not read the Excel file. Details: {e}")
        sys.exit(1)

def split_data(df: pd.DataFrame, test_zone: str) -> Tuple[List[Dict], Dict]:
    if test_zone not in df.columns:
        available_zones = [col for col in df.columns if col not in ['Ora', 'Data']]
        raise ValueError(f"The test zone '{test_zone}' is not in the dataset. Available zones: {available_zones}")
    
    training_zones = [col for col in df.columns if col not in ['Ora', 'Data', test_zone]]
    
    training_profiles = []
    for zone in training_zones:
        # Create a dictionary {hour: price} for each training zone
        profile = {int(row['Ora']) - 1: row[zone] for _, row in df.iterrows()}
        training_profiles.append(profile)
        
    test_profile = {int(row['Ora']) - 1: row[test_zone] for _, row in df.iterrows()}
    
    print(f"\nData split: {len(training_profiles)} profiles for training, 1 profile ('{test_zone}') for testing.")
    return training_profiles, test_profile

def save_results_to_excel(results_df: pd.DataFrame, output_dir: str):
    output_path = os.path.join(output_dir, "Risultati_V2G_RL_Only.xlsx")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with pd.ExcelWriter(output_path) as writer:
        results_df.to_excel(writer, sheet_name='Reinforcement Learning', index=False)
        
    print(f"\nDetailed results saved to: {output_path}")

# ========================================================================
# MAIN EXECUTION
# ========================================================================

def main():
    # 1. LOAD AND PREPARE DATA
    all_price_data = load_price_data()
    # We use other zones for training and the 'Italia' zone for testing
    training_profiles, test_profile = split_data(all_price_data, test_zone="Italia")
    
    # 2. INITIALIZE OPTIMIZER
    optimizer = V2GOptimizer(VEHICLE_PARAMS, SIMULATION_PARAMS)

    # 3. TRAIN OR LOAD THE RL AGENT
    q_table_file = RL_PARAMS['q_table_file']
    if os.path.exists(q_table_file):
        print(f"\nLoading pre-trained Q-table from '{q_table_file}'...")
        q_table = np.load(q_table_file)
    else:
        print("\nNo Q-table found. Starting RL training...")
        q_table = optimizer.train_rl_agent(RL_PARAMS, training_profiles)

    # 4. EVALUATE THE RL STRATEGY ON THE TEST DAY
    print("\n" + "="*80)
    print(f"STARTING EVALUATION ON TEST DAY (Zone: Italia)")
    print("="*80)
    
    optimizer.set_prices_for_simulation(test_profile)
    rl_results = optimizer.run_rl_strategy(q_table)
    
    # 5. SAVE RESULTS
    output_dir = os.path.dirname(RL_PARAMS['q_table_file'])
    if not output_dir: output_dir = 'output'
    save_results_to_excel(rl_results, output_dir)

if __name__ == "__main__":
    main()
