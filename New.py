# ========================================================================
# V2G OPTIMIZATION SIMULATOR
# Author: Angelo Caravella
# Version: Academic v2.4
# Description: Questo script simula e confronta diverse strategie per
#              l'ottimizzazione della carica e scarica di un veicolo
#              elettrico (Vehicle-to-Grid, V2G). Implementa euristiche
#              classiche, Model Predictive Control (MPC) e Reinforcement
#              Learning (RL), incorporando modelli di degradazione della
#              batteria e preferenze utente basati sulla letteratura
#              accademica.
#
# NOVITÀ v2.4: Corretto un bug critico di inizializzazione (AttributeError)
#              nel costruttore della classe V2GOptimizer.
#
# Riferimenti Accademici Chiave:
# 1. Ortega-Vazquez, M. A. (2014). "Optimal scheduling of electric
#    vehicle charging and vehicle-to-grid services at household level
#    including battery degradation and price uncertainty".
#
# 2. Hermans, B. A. L. M., et al. (2024). "Model predictive control of
#    vehicle charging stations in grid-connected microgrids".
#
# 3. Abdullah, H. M., et al. (2021). "Reinforcement Learning Based EV
#    Charging Management Systems-A Review".
#
# 4. Lee, S. T., et al. (2018). "A User-centric Control Scheme for
#    Residential V2G Operation to Mitigate Range Anxiety".
# ========================================================================

import numpy as np
import pandas as pd
import random
import os
import sys
from scipy.optimize import minimize, Bounds
from typing import Dict, List, Tuple

# ========================================================================
# CONFIGURAZIONE GLOBALE
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
    'mpc_horizon': 6,
    'soc_target_finale': 0.5,
}

RL_PARAMS = {
    'states_ora': 24,
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
# CLASSE MODELLO DI DEGRADAZIONE BATTERIA
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
# CLASSE OTTIMIZZATORE V2G
# ========================================================================

class V2GOptimizer:
    def __init__(self, vehicle_config: Dict, sim_config: Dict):
        # --- CORREZIONE DELL'ERRORE: Assegnazione dei parametri all'inizio ---
        self.vehicle_params = vehicle_config
        self.sim_params = sim_config
        # --- FINE CORREZIONE ---
        
        self.actions_map = {0: 'Attesa', 1: 'Carica', 2: 'Scarica'}
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
                         anxiety_cost: float, net_gain: float, reward_col_name: str):
        variation_soc = (soc_end - soc_start) * 100
        log.append({
            'Ora': hour + 1,
            'SoC Iniziale (%)': round(soc_start * 100, 1),
            'Prezzo (€/kWh)': round(price, 4),
            'Azione': action,
            'Variazione SoC (%)': round(variation_soc, 1),
            'Costo Energia (€)': round(energy_cost, 4),
            'Ricavo Energia (€)': round(energy_revenue, 4),
            'Costo Degradazione (€)': round(degradation_cost, 4),
            'Costo Ansia (€)': round(anxiety_cost, 4),
            reward_col_name: round(net_gain, 4)
        })
        print(f"{hour+1:3d} | {soc_start*100:11.1f}% | {price:6.4f} | {action:6} | "
              f"{variation_soc:13.1f} | {energy_cost:12.4f} | {energy_revenue:14.4f} | "
              f"{degradation_cost:10.4f} | {anxiety_cost:5.3f} | {net_gain:12.4f}")

    def _run_simulation_loop(self, strategy_logic):
        soc = self.sim_params['initial_soc']
        log = []
        cumulative_costs = {'energy': 0, 'revenue': 0, 'degradation': 0, 'anxiety': 0}

        for hour, price in self.current_prices.items():
            if hour >= 24: continue

            soc_start = soc
            action, power_kwh = strategy_logic(soc, hour, price)
            
            energy_cost, energy_revenue, degradation_cost = 0, 0, 0
            soc_end = soc_start
            energy_processed_kwh = 0

            if action == 'Carica':
                energy_stored_kwh = power_kwh * self.vehicle_params['efficienza_carica']
                soc_end += energy_stored_kwh / self.vehicle_params['capacita']
                energy_cost = price * power_kwh
                energy_processed_kwh = energy_stored_kwh
            elif action == 'Scarica':
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
                                  degradation_cost, anxiety_cost, net_gain, 'Guadagno Netto Ora (€)')
        
        final_soc_cost_summary = self._calculate_terminal_soc_cost(soc)
        cumulative_costs['anxiety'] += final_soc_cost_summary
        
        return self._finalize_log(log, cumulative_costs, 'Guadagno Netto Ora (€)')

    def run_heuristic_strategy(self) -> pd.DataFrame:
        print("\nDettaglio Orario - Strategia Euristica (Threshold-based):")
        print("Ora | SoC Iniziale | Prezzo | Azione | Variazione SoC | Costo Energia | Ricavo Energia | Degradazione | Ansia | Guadagno Netto")
        print("----|--------------|--------|--------|---------------|---------------|----------------|--------------|-------|---------------")
        
        prices_24h = [p for h, p in self.current_prices.items() if h < 24]
        avg_price = np.mean(prices_24h)
        std_price = np.std(prices_24h)
        charge_threshold = avg_price - 0.5 * std_price
        discharge_threshold = avg_price + 0.5 * std_price

        def logic(soc, hour, price):
            action, power_kwh = 'Attesa', 0
            if price < charge_threshold and soc < self.vehicle_params['soc_max']:
                action = 'Carica'
                power_kwh = self.vehicle_params['p_carica']
            elif price > discharge_threshold and soc > self.vehicle_params['soc_min_batteria']:
                action = 'Scarica'
                power_kwh = self.vehicle_params['p_scarica']
            return action, power_kwh

        return self._run_simulation_loop(logic)

    def run_lcvf_strategy(self) -> pd.DataFrame:
        print("\nDettaglio Orario - Strategia LCVF (Peak-Shaving/Valley-Filling Pianificato):")
        print("Ora | SoC Iniziale | Prezzo | Azione | Variazione SoC | Costo Energia | Ricavo Energia | Degradazione | Ansia | Guadagno Netto")
        print("----|--------------|--------|--------|---------------|---------------|----------------|--------------|-------|---------------")
        
        num_hours = 4
        prices_24h = {h: p for h, p in self.current_prices.items() if h < 24}
        sorted_prices = sorted(prices_24h.items(), key=lambda item: item[1])
        charge_hours = {h for h, p in sorted_prices[:num_hours]}
        discharge_hours = {h for h, p in sorted_prices[-num_hours:]}

        def logic(soc, hour, price):
            action, power_kwh = 'Attesa', 0
            if hour in charge_hours and soc < self.vehicle_params['soc_max']:
                action = 'Carica'
                power_kwh = self.vehicle_params['p_carica']
            elif hour in discharge_hours and soc > self.vehicle_params['soc_min_batteria']:
                action = 'Scarica'
                power_kwh = self.vehicle_params['p_scarica']
            return action, power_kwh

        return self._run_simulation_loop(logic)

    def run_mpc_strategy(self, horizon: int, objective_type: str = 'cost', pv_forecast: List[float] = None) -> pd.DataFrame:
        print(f"\nDettaglio Orario - Strategia MPC (Orizzonte={horizon}h, Obiettivo='{objective_type}'):")
        print("Ora | SoC Iniziale | Prezzo | Azione | Variazione SoC | Costo Energia | Ricavo Energia | Degradazione | Ansia | Guadagno Netto")
        print("----|--------------|--------|--------|---------------|---------------|----------------|--------------|-------|---------------")

        def logic(soc, hour, price):
            hours = sorted([h for h in self.current_prices.keys() if h < 24])
            current_hour_index = hours.index(hour)
            
            horizon_prices = [self.current_prices[h] for h in hours[current_hour_index : current_hour_index + horizon]]
            horizon_pv = pv_forecast[current_hour_index : current_hour_index + horizon] if pv_forecast else [0] * len(horizon_prices)
            is_terminal_for_mpc = (current_hour_index + horizon >= len(hours))

            action_power = 0
            if len(horizon_prices) > 0:
                action_power = self._solve_mpc(soc, horizon_prices, horizon_pv, is_terminal_for_mpc, objective_type)
            
            action, power_kwh = 'Attesa', 0
            if action_power > 0.1:
                action = 'Carica'
                power_kwh = action_power
            elif action_power < -0.1:
                action = 'Scarica'
                power_kwh = -action_power
            
            return action, power_kwh

        return self._run_simulation_loop(logic)

    def _solve_mpc(self, current_soc: float, horizon_prices: List[float], horizon_pv: List[float], is_terminal: bool, objective_type: str) -> float:
        n = len(horizon_prices)
        
        def objective(x_power):
            total_objective_value = 0
            soc_path = np.zeros(n + 1)
            soc_path[0] = current_soc

            for i in range(n):
                power = x_power[i]
                energy_processed_kwh = 0
                
                if power > 0:
                    energy_stored = power * self.vehicle_params['efficienza_carica']
                    soc_path[i+1] = soc_path[i] + energy_stored / self.vehicle_params['capacita']
                    energy_processed_kwh = energy_stored
                else:
                    energy_drawn = -power / self.vehicle_params['efficienza_scarica']
                    soc_path[i+1] = soc_path[i] - energy_drawn / self.vehicle_params['capacita']
                    energy_processed_kwh = -energy_drawn
                
                if objective_type == 'cost':
                    cost = 0
                    if power > 0: cost += horizon_prices[i] * power
                    else: cost -= horizon_prices[i] * -power
                    cost += self._calculate_degradation_cost(soc_path[i], soc_path[i+1], energy_processed_kwh)
                    total_objective_value += cost
                elif objective_type == 'peak_shaving':
                    grid_power = power - horizon_pv[i]
                    total_objective_value += grid_power**2
                
                total_objective_value += self._calculate_anxiety_cost(soc_path[i+1])

            if is_terminal:
                total_objective_value += self._calculate_terminal_soc_cost(soc_path[-1])
            
            return total_objective_value

        bounds = Bounds([-self.vehicle_params['p_scarica']] * n, [self.vehicle_params['p_carica']] * n)
        x0 = np.zeros(n)
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, options={'maxiter': 200})
        
        return res.x[0] if res.success else 0

    def train_rl_agent(self, rl_params: Dict, training_price_profiles: List[Dict[int, float]]) -> np.ndarray:
        print(f"\nAddestramento agente RL su {len(training_price_profiles)} profili di prezzo ({rl_params['episodes']} episodi)...")
        q_table = np.zeros((rl_params['states_ora'], rl_params['states_soc'], len(self.actions_map)))
        epsilon = rl_params['epsilon']
        
        for episode in range(rl_params['episodes']):
            episode_prices = random.choice(training_price_profiles)
            soc = self.sim_params['initial_soc']
            for ora in range(rl_params['states_ora']):
                soc_discrete = self._discretize_soc(soc, rl_params['states_soc'])
                
                if random.random() < epsilon:
                    azione = random.randint(0, len(self.actions_map) - 1)
                else:
                    azione = np.argmax(q_table[ora, soc_discrete])
                
                new_soc, reward = self._get_rl_step_result(soc, ora, azione, episode_prices, last_hour=(ora == rl_params['states_ora'] - 1))
                new_soc_discrete = self._discretize_soc(new_soc, rl_params['states_soc'])
                
                next_q_value = 0
                if ora < rl_params['states_ora'] - 1:
                    next_q_value = np.max(q_table[ora + 1, new_soc_discrete])
                
                current_q = q_table[ora, soc_discrete, azione]
                new_q = (1 - rl_params['alpha']) * current_q + rl_params['alpha'] * (reward + rl_params['gamma'] * next_q_value)
                q_table[ora, soc_discrete, azione] = new_q
                soc = new_soc
            
            epsilon = max(rl_params['epsilon_min'], epsilon * rl_params['epsilon_decay'])
            if (episode + 1) % 20000 == 0:
                print(f"Episodio {episode + 1}/{rl_params['episodes']}, Epsilon: {epsilon:.4f}")
                
        print("Addestramento RL completato!")
        q_table_dir = os.path.dirname(rl_params['q_table_file'])
        if q_table_dir and not os.path.exists(q_table_dir):
            os.makedirs(q_table_dir)
        np.save(rl_params['q_table_file'], q_table)
        print(f"Q-table salvata in '{rl_params['q_table_file']}'")
        return q_table

    def _get_rl_step_result(self, soc: float, ora: int, azione: int, prices: Dict[int, float], last_hour: bool) -> Tuple[float, float]:
        new_soc = soc
        reward = 0.0
        energy_processed_kwh = 0
        
        if azione == 1 and soc < self.vehicle_params['soc_max']:
            energy_stored = self.vehicle_params['p_carica'] * self.vehicle_params['efficienza_carica']
            new_soc += energy_stored / self.vehicle_params['capacita']
            reward -= prices.get(ora, 0) * self.vehicle_params['p_carica']
            energy_processed_kwh = energy_stored
        elif azione == 2 and soc > self.vehicle_params['soc_min_batteria']:
            energy_drawn = self.vehicle_params['p_scarica'] / self.vehicle_params['efficienza_scarica']
            new_soc -= energy_drawn / self.vehicle_params['capacita']
            reward += prices.get(ora, 0) * self.vehicle_params['p_scarica']
            energy_processed_kwh = -energy_drawn
        
        reward -= self._calculate_degradation_cost(soc, new_soc, energy_processed_kwh)
        reward -= self._calculate_anxiety_cost(new_soc)
        if last_hour:
            reward -= self._calculate_terminal_soc_cost(new_soc)
        
        return np.clip(new_soc, 0, 1), reward

    def run_rl_strategy(self, q_table: np.ndarray) -> pd.DataFrame:
        print("\nDettaglio Orario - Strategia Reinforcement Learning:")
        print("Ora | SoC Iniziale | Prezzo | Azione | Variazione SoC | Costo Energia | Ricavo Energia | Degradazione | Ansia | Guadagno Netto")
        print("----|--------------|--------|--------|---------------|---------------|----------------|--------------|-------|---------------")
        
        def logic(soc, hour, price):
            soc_discrete = self._discretize_soc(soc, q_table.shape[1])
            best_actions = np.where(q_table[hour, soc_discrete] == np.max(q_table[hour, soc_discrete]))[0]
            azione = np.random.choice(best_actions)
            
            action_str = self.actions_map[azione]
            power_kwh = 0
            if action_str == 'Carica':
                power_kwh = self.vehicle_params['p_carica']
            elif action_str == 'Scarica':
                power_kwh = self.vehicle_params['p_scarica']
            
            return action_str, power_kwh

        return self._run_simulation_loop(logic)

    def _discretize_soc(self, soc: float, states: int) -> int:
        return int(np.clip(round(soc * (states - 1)), 0, states - 1))

    def _finalize_log(self, log: List, cumulative_costs: Dict, reward_col_name: str) -> pd.DataFrame:
        total_gain = cumulative_costs['revenue'] - cumulative_costs['energy'] - cumulative_costs['degradation'] - cumulative_costs['anxiety']
        print("\nRIEPILOGO GIORNALIERO:")
        print(f"  - Costo Energia Totale: {cumulative_costs['energy']:.4f} €")
        print(f"  - Ricavo Energia Totale: {cumulative_costs['revenue']:.4f} €")
        print(f"  - Costo Degradazione Totale: {cumulative_costs['degradation']:.4f} €")
        print(f"  - Costo Ansia Totale: {cumulative_costs['anxiety']:.4f} €")
        print(f"  - GUADAGNO NETTO TOTALE: {total_gain:.4f} €")
        
        if not log: return pd.DataFrame()
        
        summary_row = {col: '' for col in log[0].keys()}
        summary_row.update({
            'Ora': 'TOTALE',
            'Costo Energia (€)': round(cumulative_costs['energy'], 4),
            'Ricavo Energia (€)': round(cumulative_costs['revenue'], 4),
            'Costo Degradazione (€)': round(cumulative_costs['degradation'], 4),
            'Costo Ansia (€)': round(cumulative_costs['anxiety'], 4),
            reward_col_name: round(total_gain, 4)
        })
        log.append(summary_row)
        return pd.DataFrame(log)

# ========================================================================
# FUNZIONI AUSILIARIE
# ========================================================================

def load_price_data(file_path: str = None) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("CARICAMENTO DATI PREZZI")
    print("=" * 80)
    
    script_dir = os.path.dirname(__file__) if '__file__' in locals() else os.getcwd()
    default_path = os.path.join(script_dir, "downloads", "PrezziZonali.xlsx")
    file_path = file_path if file_path else default_path
    
    try:
        df = pd.read_excel(file_path)
        print(f"File '{file_path}' caricato con successo.")
        
        if 'Ora' not in df.columns:
            raise ValueError("Il file Excel deve contenere una colonna 'Ora'.")
        
        for col in df.columns:
            if col not in ['Ora', 'Data']:
                try:
                    if df[col].dtype == 'object':
                        df[col] = df[col].str.replace(',', '.', regex=False).astype(float)
                    df[col] = df[col] / 1000
                except (ValueError, AttributeError) as e:
                    print(f"Attenzione: impossibile convertire la colonna '{col}'. Sarà ignorata. Errore: {e}")
                    df.drop(columns=[col], inplace=True)
        
        return df

    except FileNotFoundError:
        print(f"ERRORE: Il file '{file_path}' non è stato trovato.")
        sys.exit(1)
    except Exception as e:
        print(f"ERRORE: Impossibile leggere il file Excel. Dettagli: {e}")
        sys.exit(1)

def split_data(df: pd.DataFrame, test_zone: str) -> Tuple[List[Dict], Dict]:
    if test_zone not in df.columns:
        available_zones = [col for col in df.columns if col not in ['Ora', 'Data']]
        raise ValueError(f"La zona di test '{test_zone}' non è presente nel dataset. Zone disponibili: {available_zones}")
    
    training_zones = [col for col in df.columns if col not in ['Ora', 'Data', test_zone]]
    
    training_profiles = []
    for zone in training_zones:
        profile = {int(row['Ora']) - 1: row[zone] for _, row in df.iterrows()}
        training_profiles.append(profile)
        
    test_profile = {int(row['Ora']) - 1: row[test_zone] for _, row in df.iterrows()}
    
    print(f"\nDati divisi: {len(training_profiles)} profili per il training, 1 profilo ('{test_zone}') per il test.")
    return training_profiles, test_profile

def compare_strategies(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    comparison_data = []
    for strategy, df in results.items():
        if df.empty or len(df) < 2: continue
        summary_row = df.iloc[-1]
        reward_col = 'Guadagno Netto Ora (€)'
        
        last_op_row = df.iloc[-2]
        soc_initial_for_final = last_op_row.get('SoC Iniziale (%)', np.nan)
        soc_variation = last_op_row.get('Variazione SoC (%)', 0)
        
        soc_final = (soc_initial_for_final + soc_variation) if not np.isnan(soc_initial_for_final) else np.nan

        comparison_data.append({
            'Strategia': strategy,
            'Guadagno Netto (€)': summary_row[reward_col],
            'SOC Finale (%)': round(soc_final, 1) if not np.isnan(soc_final) else 'N/A',
            'Costo Degradazione (€)': summary_row['Costo Degradazione (€)'],
            'Costo Ansia (€)': summary_row['Costo Ansia (€)']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + "=" * 80)
    print("CONFRONTO PRESTAZIONI STRATEGIE")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print("=" * 80)
    return comparison_df

def save_results_to_excel(results: Dict, comparison_df: pd.DataFrame, output_dir: str):
    output_path = os.path.join(output_dir, "Risultati_V2G_Confronto_MultiDay.xlsx")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with pd.ExcelWriter(output_path) as writer:
        comparison_df.to_excel(writer, sheet_name='Confronto', index=False)
        for name, df in results.items():
            if not df.empty:
                df.to_excel(writer, sheet_name=name, index=False)
        
    print(f"\nRisultati dettagliati e confronto salvati in: {output_path}")

# ========================================================================
# ESECUZIONE PRINCIPALE
# ========================================================================

def main():
    # 1. CARICAMENTO E PREPARAZIONE DATI
    all_price_data = load_price_data()
    training_profiles, test_profile = split_data(all_price_data, test_zone="Italia")
    
    # 2. INIZIALIZZAZIONE
    optimizer = V2GOptimizer(VEHICLE_PARAMS, SIMULATION_PARAMS)
    results = {}

    # 3. TRAINING DELL'AGENTE RL (se necessario)
    q_table_file = RL_PARAMS['q_table_file']
    if os.path.exists(q_table_file):
        print(f"\nCaricamento Q-table pre-addestrata da '{q_table_file}'...")
        q_table = np.load(q_table_file)
    else:
        print("\nNessuna Q-table trovata. Inizio addestramento RL...")
        q_table = optimizer.train_rl_agent(RL_PARAMS, training_profiles)

    # 4. VALUTAZIONE DI TUTTE LE STRATEGIE SULLA GIORNATA DI TEST
    print("\n" + "="*80)
    print(f"INIZIO VALUTAZIONE SULLA GIORNATA DI TEST (Zona: Italia)")
    print("="*80)
    
    optimizer.set_prices_for_simulation(test_profile)

    print("\n>>> STRATEGIA EURISTICA SEMPLICE <<<")
    results['Euristica Semplice'] = optimizer.run_heuristic_strategy()

    print("\n>>> STRATEGIA EURISTICA LCVF <<<")
    results['Euristica LCVF'] = optimizer.run_lcvf_strategy()
    
    print("\n>>> STRATEGIA MPC (COSTO) <<<")
    results[f"MPC (O={SIMULATION_PARAMS['mpc_horizon']}h, Costo)"] = optimizer.run_mpc_strategy(
        horizon=SIMULATION_PARAMS['mpc_horizon'], 
        objective_type='cost'
    )
    
    print("\n>>> STRATEGIA REINFORCEMENT LEARNING <<<")
    results['Reinforcement Learning'] = optimizer.run_rl_strategy(q_table)
    
    # 5. CONFRONTO FINALE E SALVATAGGIO
    comparison_df = compare_strategies(results)
    
    output_dir = os.path.dirname(RL_PARAMS['q_table_file'])
    if not output_dir: output_dir = 'output'
    save_results_to_excel(results, comparison_df, output_dir)

if __name__ == "__main__":
    main()
