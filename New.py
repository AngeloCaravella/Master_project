# ========================================================================
# V2G INTERACTIVE ANALYSIS DASHBOARD
# Author: Angelo Caravella 
# Version: 3.0
# Description: Versione finale con logica di configurazione corretta (SoC
#              iniziale dinamico) e caricamento di agenti specializzati.
#
# Requisiti Aggiuntivi:
# pip install torch
# ========================================================================

import numpy as np
import pandas as pd
import random
import os
import sys
from scipy.optimize import minimize, Bounds
from typing import Dict, List, Tuple

# Tentativo di importare PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ========================================================================
# CONFIGURAZIONI PREDEFINITE
# ========================================================================

USER_PROFILES = {
    'conservativo': {
        'initial_soc': 0.70, 'soc_min_utente': 0.60, 'penalita_ansia': 0.02, 'soc_target_finale': 0.70,
    },
    'bilanciato': {
        'initial_soc': 0.50, 'soc_min_utente': 0.30, 'penalita_ansia': 0.01, 'soc_target_finale': 0.50,
    },
    'aggressivo': {
        'initial_soc': 0.20, 'soc_min_utente': 0.15, 'penalita_ansia': 0.005, 'soc_target_finale': 0.20,
    }
}

BATTERY_CHEMISTRIES = {
    'nca': {
        'degradation_model': 'nca', 'costo_batteria': 150 * 60
    },
    'lfp': {
        'degradation_model': 'lfp', 'costo_batteria': 110 * 60
    },
    'semplice': {
        'degradation_model': 'simple', 'costo_batteria': 150 * 60
    }
}

BASE_VEHICLE_PARAMS = {
    'capacita': 60, 'p_carica': 7.4, 'p_scarica': 5.0, 'efficienza_carica': 0.95,
    'efficienza_scarica': 0.95, 'soc_max': 0.9, 'soc_min_batteria': 0.1, 'lfp_k_slope': 0.0035,
}

BASE_SIMULATION_PARAMS = {
    'mpc_horizon': 12,
}

RL_PARAMS = {
    'states_ora': 24, 'states_soc': 11
}

DQN_PARAMS = {
    'state_size': 3, 'action_size': 3, 'hidden_size': 64,
}

# ========================================================================
# CLASSI (DQN, Degradation, Optimizer) - (Invariate)
# ========================================================================

if TORCH_AVAILABLE:
    class QNetwork(nn.Module):
        def __init__(self, state_size: int, action_size: int, hidden_size: int):
            super(QNetwork, self).__init__()
            self.fc1 = nn.Linear(state_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, action_size)
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)

    class DQNAgentForEval:
        def __init__(self):
            self.state_size = DQN_PARAMS['state_size']
            self.action_size = DQN_PARAMS['action_size']
            self.policy_net = QNetwork(self.state_size, self.action_size, DQN_PARAMS['hidden_size'])
        def choose_action(self, state: np.ndarray) -> int:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            return torch.argmax(q_values).item()
        def load_model(self, path: str):
            self.policy_net.load_state_dict(torch.load(path))
            self.policy_net.eval()
            print(f"Modello DQN caricato da '{path}'")

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

class V2GOptimizer:
    def __init__(self, vehicle_config: Dict, sim_config: Dict):
        self.vehicle_params = vehicle_config
        self.sim_params = sim_config
        self.actions_map = {0: 'Attesa', 1: 'Carica', 2: 'Scarica'}
        self.degradation_model_type = self.vehicle_params.get('degradation_model', 'simple')
        self.degradation_calculator = BatteryDegradationModel(self.vehicle_params)
        self.current_prices = None
    def set_prices_for_simulation(self, prices: Dict[int, float]):
        self.current_prices = prices
    def _calculate_degradation_cost(self, soc_start: float, soc_end: float, energy_kwh: float) -> float:
        if self.degradation_model_type == 'lfp': return self.degradation_calculator.cost_lfp_model(energy_kwh)
        elif self.degradation_model_type == 'nca': return self.degradation_calculator.cost_nca_model(soc_start, soc_end)
        else: return self.degradation_calculator.cost_simple_linear(energy_kwh)
    def _calculate_anxiety_cost(self, soc: float) -> float:
        if soc < self.sim_params['soc_min_utente']: return self.sim_params['penalita_ansia'] * (self.sim_params['soc_min_utente'] - soc) * 100
        return 0.0
    def _calculate_terminal_soc_cost(self, soc: float) -> float:
        target = self.sim_params['soc_target_finale']
        if soc < target: return self.sim_params['penalita_ansia'] * 5 * (target - soc) * 100
        return 0.0
    def _run_simulation_loop(self, strategy_logic):
        soc = self.sim_params['initial_soc']
        log = []
        for hour, price in self.current_prices.items():
            if hour >= 24: continue
            soc_start = soc
            action, power_kwh = strategy_logic(soc, hour, price)
            energy_cost, energy_revenue, degradation_cost, energy_processed_kwh = 0, 0, 0, 0
            soc_end = soc_start
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
            soc = np.clip(soc_end, 0, 1)
            log.append({'Ora': hour + 1, 'SoC Iniziale (%)': soc_start * 100, 'Variazione SoC (%)': (soc - soc_start) * 100,
                        'Costo Energia (€)': energy_cost, 'Ricavo Energia (€)': energy_revenue,
                        'Costo Degradazione (€)': degradation_cost, 'Costo Ansia (€)': anxiety_cost, 'Azione': action})
        terminal_cost = self._calculate_terminal_soc_cost(soc)
        if log: log[-1]['Costo Ansia (€)'] += terminal_cost
        return pd.DataFrame(log)
    def run_heuristic_strategy(self) -> pd.DataFrame:
        prices_24h = [p for h, p in self.current_prices.items() if h < 24]
        avg_price, std_price = np.mean(prices_24h), np.std(prices_24h)
        charge_threshold, discharge_threshold = avg_price - 0.5 * std_price, avg_price + 0.5 * std_price
        def logic(soc, hour, price):
            action, power_kwh = 'Attesa', 0
            if price < charge_threshold and soc < self.vehicle_params['soc_max']: action, power_kwh = 'Carica', self.vehicle_params['p_carica']
            elif price > discharge_threshold and soc > self.vehicle_params['soc_min_batteria']: action, power_kwh = 'Scarica', self.vehicle_params['p_scarica']
            return action, power_kwh
        return self._run_simulation_loop(logic)
    def run_lcvf_strategy(self) -> pd.DataFrame:
        num_hours = 4
        prices_24h = {h: p for h, p in self.current_prices.items() if h < 24}
        sorted_prices = sorted(prices_24h.items(), key=lambda item: item[1])
        charge_hours, discharge_hours = {h for h, p in sorted_prices[:num_hours]}, {h for h, p in sorted_prices[-num_hours:]}
        def logic(soc, hour, price):
            action, power_kwh = 'Attesa', 0
            if hour in charge_hours and soc < self.vehicle_params['soc_max']: action, power_kwh = 'Carica', self.vehicle_params['p_carica']
            elif hour in discharge_hours and soc > self.vehicle_params['soc_min_batteria']: action, power_kwh = 'Scarica', self.vehicle_params['p_scarica']
            return action, power_kwh
        return self._run_simulation_loop(logic)
    def run_mpc_strategy(self) -> pd.DataFrame:
        def logic(soc, hour, price):
            hours = sorted([h for h in self.current_prices.keys() if h < 24])
            current_hour_index = hours.index(hour)
            horizon = self.sim_params['mpc_horizon']
            horizon_prices = [self.current_prices[h] for h in hours[current_hour_index : current_hour_index + horizon]]
            is_terminal_for_mpc = (current_hour_index + horizon >= len(hours))
            action_power = self._solve_mpc(soc, horizon_prices, is_terminal_for_mpc) if horizon_prices else 0
            action, power_kwh = 'Attesa', 0
            if action_power > 0.1: action, power_kwh = 'Carica', action_power
            elif action_power < -0.1: action, power_kwh = 'Scarica', -action_power
            return action, power_kwh
        return self._run_simulation_loop(logic)
    def _solve_mpc(self, current_soc: float, horizon_prices: List[float], is_terminal: bool) -> float:
        n = len(horizon_prices)
        def objective(x_power):
            total_objective_value = 0
            soc_path = np.zeros(n + 1)
            soc_path[0] = current_soc
            for i in range(n):
                power, energy_processed_kwh = x_power[i], 0
                if power > 0:
                    energy_stored = power * self.vehicle_params['efficienza_carica']
                    soc_path[i+1] = soc_path[i] + energy_stored / self.vehicle_params['capacita']
                    energy_processed_kwh = energy_stored
                else:
                    energy_drawn = -power / self.vehicle_params['efficienza_scarica']
                    soc_path[i+1] = soc_path[i] - energy_drawn / self.vehicle_params['capacita']
                    energy_processed_kwh = -energy_drawn
                cost = (horizon_prices[i] * power) + self._calculate_degradation_cost(soc_path[i], soc_path[i+1], energy_processed_kwh)
                total_objective_value += cost + self._calculate_anxiety_cost(soc_path[i+1])
            if is_terminal: total_objective_value += self._calculate_terminal_soc_cost(soc_path[-1])
            return total_objective_value
        bounds = Bounds([-self.vehicle_params['p_scarica']] * n, [self.vehicle_params['p_carica']] * n)
        res = minimize(objective, np.zeros(n), method='SLSQP', bounds=bounds, options={'maxiter': 200})
        return res.x[0] if res.success else 0
    def run_rl_strategy(self, q_table: np.ndarray) -> pd.DataFrame:
        def logic(soc, hour, price):
            soc_discrete = int(np.clip(round(soc * (RL_PARAMS['states_soc'] - 1)), 0, RL_PARAMS['states_soc'] - 1))
            azione = np.argmax(q_table[hour, soc_discrete])
            action_str, power_kwh = self.actions_map[azione], 0
            if action_str == 'Carica': power_kwh = self.vehicle_params['p_carica']
            elif action_str == 'Scarica': power_kwh = self.vehicle_params['p_scarica']
            return action_str, power_kwh
        return self._run_simulation_loop(logic)
    def run_dqn_strategy(self, agent) -> pd.DataFrame:
        max_price_for_norm = 0.5
        def logic(soc, hour, price):
            norm_hour, norm_price = hour / 23.0, min(price / max_price_for_norm, 1.0)
            state = np.array([norm_hour, soc, norm_price], dtype=np.float32)
            action_idx = agent.choose_action(state)
            action_str, power_kwh = self.actions_map[action_idx], 0
            if action_str == 'Carica': power_kwh = self.vehicle_params['p_carica']
            elif action_str == 'Scarica': power_kwh = self.vehicle_params['p_scarica']
            return action_str, power_kwh
        return self._run_simulation_loop(logic)

# ========================================================================
# FUNZIONI AUSILIARIE
# ========================================================================

def load_price_data(file_path: str = "downloads/PrezziZonali.xlsx") -> pd.DataFrame:
    try:
        df = pd.read_excel(file_path)
        for col in df.columns:
            if col not in ['Ora', 'Data']:
                if df[col].dtype == 'object': df[col] = df[col].str.replace(',', '.', regex=False).astype(float)
                df[col] = df[col] / 1000
        return df
    except FileNotFoundError: print(f"ERRORE: File prezzi '{file_path}' non trovato.", file=sys.stderr); sys.exit(1)

def create_daily_profiles(df: pd.DataFrame, test_zone: str = "Italia") -> Tuple[List[Dict], List[Dict]]:
    all_zones = [col for col in df.columns if col not in ['Ora', 'Data']]
    training_zones = [z for z in all_zones if z != test_zone]
    training_profiles, test_profiles = [], []
    for zone in training_zones:
        zone_prices = df[zone].dropna()
        for i in range(0, len(zone_prices), 24):
            if len(zone_prices.iloc[i:i+24]) == 24: training_profiles.append({h: p for h, p in enumerate(zone_prices.iloc[i:i+24])})
    if test_zone in df.columns:
        test_prices = df[test_zone].dropna()
        for i in range(0, len(test_prices), 24):
            if len(test_prices.iloc[i:i+24]) == 24: test_profiles.append({h: p for h, p in enumerate(test_prices.iloc[i:i+24])})
    print(f"Dati processati: {len(training_profiles)} giorni per training, {len(test_profiles)} per test.")
    return training_profiles, test_profiles

def validate_config(vehicle_params, sim_params):
    if not (0 <= vehicle_params['soc_min_batteria'] < sim_params['soc_min_utente'] < vehicle_params['soc_max'] <= 1):
        raise ValueError("Vincolo logico violato: 0 <= soc_min_batteria < soc_min_utente < soc_max <= 1")
    if not (vehicle_params['soc_min_batteria'] <= sim_params['initial_soc'] <= vehicle_params['soc_max']):
        raise ValueError("Vincolo logico violato: soc_min_batteria <= initial_soc <= soc_max")
    if not (vehicle_params['soc_min_batteria'] <= sim_params['soc_target_finale'] <= vehicle_params['soc_max']):
        raise ValueError("Vincolo logico violato: soc_min_batteria <= soc_target_finale <= soc_max")
    print("Configurazione valida.")

def compare_strategies(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    comparison_data = []
    for strategy, df in results.items():
        if df.empty: continue
        energy_cost, energy_revenue = df['Costo Energia (€)'].sum(), df['Ricavo Energia (€)'].sum()
        degradation_cost, anxiety_cost = df['Costo Degradazione (€)'].sum(), df['Costo Ansia (€)'].sum()
        last_row = df.iloc[-1]
        soc_final = last_row['SoC Iniziale (%)'] + last_row['Variazione SoC (%)']
        financial_gain = energy_revenue - energy_cost - degradation_cost
        comparison_data.append({'Strategia': strategy, 'Guadagno Finanziario (€)': financial_gain,
                                'Costo Virtuale (€)': anxiety_cost, 'SOC Finale (%)': round(soc_final, 1)})
    return pd.DataFrame(comparison_data).sort_values(by='Guadagno Finanziario (€)', ascending=False)

def display_profile_options(title="Scegli il profilo utente:"):
    print(f"\n{title}")
    for i, (name, params) in enumerate(USER_PROFILES.items()):
        if name == 'conservativo': desc = "Priorità alla sicurezza, inizia e finisce con molta carica."
        elif name == 'bilanciato': desc = "Equilibrio tra guadagno e salute della batteria (standard)."
        elif name == 'aggressivo': desc = "Massimizza il profitto, accettando maggior rischio."
        print(f"  {i+1}) {name.capitalize()}: {desc}")
        print(f"     (SoC Iniziale: {params['initial_soc']*100:.0f}%, SoC Min: {params['soc_min_utente']*100:.0f}%, Target Finale: {params['soc_target_finale']*100:.0f}%)")

# ========================================================================
# ESECUZIONE PRINCIPALE
# ========================================================================

def run_full_simulation(vehicle_config, sim_config):
    try: validate_config(vehicle_config, sim_config)
    except ValueError as e: print(f"ERRORE DI CONFIGURAZIONE: {e}", file=sys.stderr); return

    price_data = load_price_data()
    _, test_profiles = create_daily_profiles(price_data)
    if not test_profiles: return

    profile_name, chem_name = sim_config['profile_name'], vehicle_config['chem_name']
    q_table_path = os.path.join('q_tables', f"q_table_{profile_name}_{chem_name}.npy")
    q_table = np.load(q_table_path) if os.path.exists(q_table_path) else None
    if q_table is None: print(f"ATTENZIONE: Q-table '{q_table_path}' non trovata. Sarà saltata.")
    else: print(f"Q-table '{q_table_path}' caricata.")

    dqn_agent = None
    if TORCH_AVAILABLE:
        dqn_model_path = f"dqn_model_{profile_name}_{chem_name}.pth"
        if os.path.exists(dqn_model_path):
            dqn_agent = DQNAgentForEval()
            dqn_agent.load_model(dqn_model_path)
        else: print(f"ATTENZIONE: Modello DQN '{dqn_model_path}' non trovato. Sarà saltato.")

    test_day = random.choice(test_profiles)
    optimizer = V2GOptimizer(vehicle_config, sim_config)
    optimizer.set_prices_for_simulation(test_day)
    results = {}
    
    print("\n" + "-"*80)
    print(f"INIZIO SIMULAZIONE GIORNALIERA - Profilo: {profile_name.upper()}, Batteria: {chem_name.upper()}")
    print("-"*80)

    strategies_to_run = {"Euristica": optimizer.run_heuristic_strategy, "LCVF": optimizer.run_lcvf_strategy,
                         f"MPC (O={sim_config['mpc_horizon']}h)": optimizer.run_mpc_strategy}
    if q_table is not None: strategies_to_run[f'RL (Tab_{chem_name[:3]})'] = lambda: optimizer.run_rl_strategy(q_table)
    if dqn_agent: strategies_to_run[f'DQN ({chem_name[:3]})'] = lambda: optimizer.run_dqn_strategy(dqn_agent)

    for name, func in strategies_to_run.items():
        print(f"\n--- Dettaglio Orario Strategia: {name} ---")
        df = func()
        results[name] = df
        df_display = df.copy()
        df_display['SoC Iniziale (%)'] = df_display['SoC Iniziale (%)'].map('{:,.1f}'.format)
        df_display['Variazione SoC (%)'] = df_display['Variazione SoC (%)'].map('{:,.1f}'.format)
        print(df_display[['Ora', 'Azione', 'SoC Iniziale (%)', 'Variazione SoC (%)', 'Costo Energia (€)', 'Ricavo Energia (€)', 'Costo Degradazione (€)', 'Costo Ansia (€)']].to_string(index=False))

    print("\n" + "="*80)
    print(f"RISULTATI CONFRONTO FINALE - Profilo: {profile_name.upper()}, Batteria: {chem_name.upper()}")
    print("="*80)
    comparison_df = compare_strategies(results)
    print(comparison_df.to_string(index=False))
    print("="*80)

def main():
    while True:
        print("\n--- MENU PRINCIPALE - CRUSCOTTO DI ANALISI V2G ---")
        print("1) Esegui una singola simulazione (configurabile)")
        print("2) Esegui analisi di sensibilità (su un profilo utente)")
        print("3) Pulisci tutti i modelli e le Q-table addestrrate")
        print("4) Esci")
        choice = input("Scelta [1-4]: ").strip()

        if choice == '1':
            profile_map = {str(i+1): p for i, p in enumerate(USER_PROFILES.keys())}
            while True:
                display_profile_options()
                p_choice = input(f"Scelta [1-{len(profile_map)}]: ").strip()
                if p_choice in profile_map: profile_name = profile_map[p_choice]; break
                print("Scelta non valida.")
            
            chem_map = {str(i+1): c for i, c in enumerate(BATTERY_CHEMISTRIES.keys())}
            while True:
                print("\nScegli la chimica della batteria:")
                for i, c in chem_map.items(): print(f"  {i}) {c.upper()}")
                c_choice = input(f"Scelta [1-{len(chem_map)}]: ").strip()
                if c_choice in chem_map: chem_name = chem_map[c_choice]; break
                print("Scelta non valida.")

            sim_config = {**BASE_SIMULATION_PARAMS, **USER_PROFILES[profile_name], 'profile_name': profile_name}
            vehicle_config = {**BASE_VEHICLE_PARAMS, **BATTERY_CHEMISTRIES[chem_name], 'chem_name': chem_name}
            run_full_simulation(vehicle_config, sim_config)

        elif choice == '2':
            profile_map = {str(i+1): p for i, p in enumerate(USER_PROFILES.keys())}
            while True:
                display_profile_options("Scegli il profilo utente per l'analisi di sensibilità:")
                p_choice = input(f"Scelta [1-{len(profile_map)}]: ").strip()
                if p_choice in profile_map: profile_name = profile_map[p_choice]; break
                print("Scelta non valida.")
            
            print("\n--- AVVIO ANALISI DI SENSIBILITÀ SULLA CHIMICA DELLA BATTERIA ---")
            for chem_name in BATTERY_CHEMISTRIES.keys():
                sim_config = {**BASE_SIMULATION_PARAMS, **USER_PROFILES[profile_name], 'profile_name': profile_name}
                vehicle_config = {**BASE_VEHICLE_PARAMS, **BATTERY_CHEMISTRIES[chem_name], 'chem_name': chem_name}
                run_full_simulation(vehicle_config, sim_config)
        
        elif choice == '3':
            print("\n--- PULIZIA DEI MODELLI ADDESTRATI ---")
            confirm = input("Sei sicuro di voler eliminare tutti i file .pth e .npy? Questa azione è irreversibile. [s/N]: ").strip().lower()
            if confirm == 's':
                deleted_files = 0
                for file in os.listdir():
                    if file.endswith('.pth'): os.remove(file); deleted_files += 1
                if os.path.exists('q_tables'):
                    for file in os.listdir('q_tables'):
                        if file.endswith('.npy'): os.remove(os.path.join('q_tables', file)); deleted_files += 1
                print(f"Pulizia completata. {deleted_files} file eliminati.")
            else:
                print("Azione annullata.")

        elif choice == '4': print("Uscita."); break
        else: print("Scelta non valida.")

if __name__ == "__main__":
    main()
