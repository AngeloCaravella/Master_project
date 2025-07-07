# ========================================================================
# V2G INTERACTIVE ANALYSIS DASHBOARD
# Author: Angelo Caravella
# Version: 4.0 (Targeted Comparisons)
# Description: Ristrutturato il menu per creare confronti metodologicamente
#              corretti: Specialisti (Q-Table, DQN Spec) tra loro e con
#              le strategie classiche, e il Generalista contro le strategie
#              adattive (MPC, Euristiche).
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
# CONFIGURAZIONI PREDEFINITE (Invariate)
# ========================================================================
# ... (il codice delle configurazioni rimane identico) ...
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
    'nca': {'degradation_model': 'nca', 'costo_batteria': 120},
    'lfp': {'degradation_model': 'lfp', 'costo_batteria': 90},
    'semplice': {'degradation_model': 'simple', 'costo_batteria': 70}
}
BASE_VEHICLE_PARAMS = {
    'capacita': 60, 'p_carica': 7.4, 'p_scarica': 5.0, 'efficienza_carica': 0.95,
    'efficienza_scarica': 0.95, 'soc_max': 0.9, 'soc_min_batteria': 0.1, 'lfp_k_slope': 0.0035,
}
BASE_SIMULATION_PARAMS = {'mpc_horizon': 12}
RL_PARAMS = {'states_ora': 24, 'states_soc': 11}
DQN_PARAMS = {'state_size': 3, 'action_size': 3, 'hidden_size': 64}
DQN_GENERALIST_PARAMS = {'state_size': 5, 'action_size': 3, 'hidden_size': 128}


# ========================================================================
# CLASSI (DQN, Degradation, Optimizer) (Invariate)
# ========================================================================
# ... (il codice delle classi rimane identico) ...
if TORCH_AVAILABLE:
    class QNetwork(nn.Module):
        def __init__(self, state_size: int, action_size: int, hidden_size: int):
            super(QNetwork, self).__init__()
            self.fc1 = nn.Linear(state_size, hidden_size); self.fc2 = nn.Linear(hidden_size, hidden_size); self.fc3 = nn.Linear(hidden_size, action_size)
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = torch.relu(self.fc1(x)); x = torch.relu(self.fc2(x)); return self.fc3(x)

    class DQNAgentForEval:
        def __init__(self, state_size: int, hidden_size: int):
            self.state_size = state_size; self.action_size = 3; self.policy_net = QNetwork(self.state_size, self.action_size, hidden_size)
        def choose_action(self, state: np.ndarray) -> int:
            state_tensor = torch.FloatTensor(state).unsqueeze(0);
            with torch.no_grad(): q_values = self.policy_net(state_tensor)
            return torch.argmax(q_values).item()
        def load_model(self, path: str):
            self.policy_net.load_state_dict(torch.load(path)); self.policy_net.eval(); print(f"Modello DQN caricato da '{path}'")

class BatteryDegradationModel:
    def __init__(self, vehicle_config: Dict):
        self.vehicle_params = vehicle_config; self.battery_cost = vehicle_config['costo_batteria']; self.battery_capacity = vehicle_config['capacita']
    def _cycle_life_phi_nca(self, dod: float) -> float:
        dod_perc = dod * 100; return 0.0 if dod_perc <= 0 else 6.6e-6 * np.exp(0.045 * dod_perc)
    def cost_simple_linear(self, energy_kwh: float) -> float: return 0.008 * abs(energy_kwh)
    def cost_lfp_model(self, energy_kwh: float) -> float:
        k = self.vehicle_params['lfp_k_slope']; return (abs(energy_kwh) / self.battery_capacity) * (k / 100) * self.battery_cost
    def cost_nca_model(self, soc_start: float, soc_end: float) -> float:
        if soc_end >= soc_start: return 0.0
        dod_start = 1.0 - soc_start; dod_end = 1.0 - soc_end
        inv_phi_start = self._cycle_life_phi_nca(dod_start); inv_phi_end = self._cycle_life_phi_nca(dod_end)
        return (inv_phi_end - inv_phi_start) * self.battery_cost

class V2GOptimizer:
    def __init__(self, vehicle_config: Dict, sim_config: Dict):
        self.vehicle_params = vehicle_config; self.sim_params = sim_config; self.actions_map = {0: 'Attesa', 1: 'Carica', 2: 'Scarica'}
        self.degradation_model_type = self.vehicle_params.get('degradation_model', 'simple'); self.degradation_calculator = BatteryDegradationModel(self.vehicle_params); self.current_prices = None
    def set_prices_for_simulation(self, prices: Dict[int, float]): self.current_prices = prices
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
        soc = self.sim_params['initial_soc']; log = []
        for hour, price in self.current_prices.items():
            if hour >= 24: continue
            soc_start = soc; action, power_kwh = strategy_logic(soc, hour, price); energy_cost, energy_revenue, degradation_cost, energy_processed_kwh = 0, 0, 0, 0; soc_end = soc_start
            if action == 'Carica':
                energy_stored_kwh = power_kwh * self.vehicle_params['efficienza_carica']; soc_end += energy_stored_kwh / self.vehicle_params['capacita']
                energy_cost = price * power_kwh; energy_processed_kwh = energy_stored_kwh
            elif action == 'Scarica':
                energy_drawn_kwh = power_kwh / self.vehicle_params['efficienza_scarica']; soc_end -= energy_drawn_kwh / self.vehicle_params['capacita']
                energy_revenue = price * power_kwh; energy_processed_kwh = -energy_drawn_kwh
            degradation_cost = self._calculate_degradation_cost(soc_start, soc_end, energy_processed_kwh); anxiety_cost = self._calculate_anxiety_cost(soc_end); soc = np.clip(soc_end, 0, 1)
            log.append({'Ora': hour + 1, 'SoC Iniziale (%)': soc_start * 100, 'Variazione SoC (%)': (soc - soc_start) * 100, 'Costo Energia (€)': energy_cost, 'Ricavo Energia (€)': energy_revenue, 'Costo Degradazione (€)': degradation_cost, 'Costo Ansia (€)': anxiety_cost, 'Azione': action})
        terminal_cost = self._calculate_terminal_soc_cost(soc);
        if log: log[-1]['Costo Ansia (€)'] += terminal_cost
        return pd.DataFrame(log)
    def run_heuristic_strategy(self) -> pd.DataFrame:
        prices_24h = [p for h, p in self.current_prices.items() if h < 24]; avg_price, std_price = np.mean(prices_24h), np.std(prices_24h); charge_threshold, discharge_threshold = avg_price - 0.5 * std_price, avg_price + 0.5 * std_price
        def logic(soc, hour, price):
            action, power_kwh = 'Attesa', 0
            if price < charge_threshold and soc < self.vehicle_params['soc_max']: action, power_kwh = 'Carica', self.vehicle_params['p_carica']
            elif price > discharge_threshold and soc > self.vehicle_params['soc_min_batteria']: action, power_kwh = 'Scarica', self.vehicle_params['p_scarica']
            return action, power_kwh
        return self._run_simulation_loop(logic)
    def run_lcvf_strategy(self) -> pd.DataFrame:
        num_hours = 4; prices_24h = {h: p for h, p in self.current_prices.items() if h < 24}; sorted_prices = sorted(prices_24h.items(), key=lambda item: item[1]); charge_hours, discharge_hours = {h for h, p in sorted_prices[:num_hours]}, {h for h, p in sorted_prices[-num_hours:]}
        def logic(soc, hour, price):
            action, power_kwh = 'Attesa', 0
            if hour in charge_hours and soc < self.vehicle_params['soc_max']: action, power_kwh = 'Carica', self.vehicle_params['p_carica']
            elif hour in discharge_hours and soc > self.vehicle_params['soc_min_batteria']: action, power_kwh = 'Scarica', self.vehicle_params['p_scarica']
            return action, power_kwh
        return self._run_simulation_loop(logic)
    def run_mpc_strategy(self) -> pd.DataFrame:
        def logic(soc, hour, price):
            hours = sorted([h for h in self.current_prices.keys() if h < 24]); current_hour_index = hours.index(hour); horizon = self.sim_params['mpc_horizon']; horizon_prices = [self.current_prices[h] for h in hours[current_hour_index : current_hour_index + horizon]]; is_terminal_for_mpc = (current_hour_index + horizon >= len(hours)); action_power = self._solve_mpc(soc, horizon_prices, is_terminal_for_mpc) if horizon_prices else 0
            action, power_kwh = 'Attesa', 0
            if action_power > 0.1: action, power_kwh = 'Carica', action_power
            elif action_power < -0.1: action, power_kwh = 'Scarica', -action_power
            return action, power_kwh
        return self._run_simulation_loop(logic)
    def _solve_mpc(self, current_soc: float, horizon_prices: List[float], is_terminal: bool) -> float:
        n = len(horizon_prices)
        def objective(x_power):
            total_objective_value = 0; soc_path = np.zeros(n + 1); soc_path[0] = current_soc
            for i in range(n):
                power, energy_processed_kwh = x_power[i], 0
                if power > 0: energy_stored = power * self.vehicle_params['efficienza_carica']; soc_path[i+1] = soc_path[i] + energy_stored / self.vehicle_params['capacita']; energy_processed_kwh = energy_stored
                else: energy_drawn = -power / self.vehicle_params['efficienza_scarica']; soc_path[i+1] = soc_path[i] - energy_drawn / self.vehicle_params['capacita']; energy_processed_kwh = -energy_drawn
                cost = (horizon_prices[i] * power) + self._calculate_degradation_cost(soc_path[i], soc_path[i+1], energy_processed_kwh); total_objective_value += cost + self._calculate_anxiety_cost(soc_path[i+1])
            if is_terminal: total_objective_value += self._calculate_terminal_soc_cost(soc_path[-1])
            return total_objective_value
        bounds = Bounds([-self.vehicle_params['p_scarica']] * n, [self.vehicle_params['p_carica']] * n); res = minimize(objective, np.zeros(n), method='SLSQP', bounds=bounds, options={'maxiter': 200}); return res.x[0] if res.success else 0
    def run_rl_strategy(self, q_table: np.ndarray) -> pd.DataFrame:
        def logic(soc, hour, price):
            soc_discrete = int(np.clip(round(soc * (RL_PARAMS['states_soc'] - 1)), 0, RL_PARAMS['states_soc'] - 1)); azione = np.argmax(q_table[hour, soc_discrete]); action_str, power_kwh = self.actions_map[azione], 0
            if action_str == 'Carica': power_kwh = self.vehicle_params['p_carica']
            elif action_str == 'Scarica': power_kwh = self.vehicle_params['p_scarica']
            return action_str, power_kwh
        return self._run_simulation_loop(logic)
    def run_dqn_strategy(self, agent: DQNAgentForEval, is_generalist: bool = False) -> pd.DataFrame:
        max_price_for_norm = 0.5
        def logic(soc, hour, price):
            norm_hour, norm_price = hour / 23.0, min(price / max_price_for_norm, 1.0)
            if is_generalist: state = np.array([norm_hour, soc, norm_price, self.sim_params['soc_min_utente'], self.sim_params['soc_target_finale']], dtype=np.float32)
            else: state = np.array([norm_hour, soc, norm_price], dtype=np.float32)
            action_idx = agent.choose_action(state); action_str, power_kwh = self.actions_map[action_idx], 0
            if action_str == 'Carica': power_kwh = self.vehicle_params['p_carica']
            elif action_str == 'Scarica': power_kwh = self.vehicle_params['p_scarica']
            return action_str, power_kwh
        return self._run_simulation_loop(logic)

# ========================================================================
# FUNZIONI AUSILIARIE
# ========================================================================
def load_price_data(file_path: str = "downloads/PrezziZonali.xlsx") -> pd.DataFrame:
    try: df = pd.read_excel(file_path)
    except FileNotFoundError: print(f"ERRORE: File prezzi '{file_path}' non trovato.", file=sys.stderr); sys.exit(1)
    for col in df.columns:
        if col not in ['Ora', 'Data'] and df[col].dtype == 'object': df[col] = df[col].str.replace(',', '.', regex=False).astype(float)
        if col not in ['Ora', 'Data']: df[col] = df[col] / 1000
    return df

def create_test_profiles(df: pd.DataFrame, test_zone: str = "Italia") -> List[Dict]:
    test_profiles = []
    if test_zone in df.columns:
        zone_prices = df[test_zone].dropna()
        for i in range(0, len(zone_prices), 24):
            if len(zone_prices.iloc[i:i+24]) == 24: test_profiles.append({h: p for h, p in enumerate(zone_prices.iloc[i:i+24])})
    print(f"Dati di test processati: {len(test_profiles)} giorni trovati per la zona '{test_zone}'.")
    return test_profiles

def validate_config(sim_params, vehicle_params):
    min_batt, max_batt = vehicle_params['soc_min_batteria'], vehicle_params['soc_max']
    min_user, target_user, initial_soc = sim_params['soc_min_utente'], sim_params['soc_target_finale'], sim_params['initial_soc']
    if not (min_batt < min_user < max_batt): raise ValueError(f"SoC Min Utente ({min_user*100:.0f}%) deve essere tra i limiti fisici ({min_batt*100:.0f}%-{max_batt*100:.0f}%).")
    if not (min_batt <= initial_soc <= max_batt): raise ValueError(f"SoC Iniziale ({initial_soc*100:.0f}%) deve essere tra i limiti fisici.")
    if not (min_batt <= target_user <= max_batt): raise ValueError(f"SoC Target ({target_user*100:.0f}%) deve essere tra i limiti fisici.")
    print("Configurazione valida.")
    return True

def compare_strategies(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    comparison_data = []
    for strategy, df in results.items():
        if df.empty: continue
        energy_cost, energy_revenue = df['Costo Energia (€)'].sum(), df['Ricavo Energia (€)'].sum()
        degradation_cost, anxiety_cost = df['Costo Degradazione (€)'].sum(), df['Costo Ansia (€)'].sum()
        last_row = df.iloc[-1]; soc_final = last_row['SoC Iniziale (%)'] + last_row['Variazione SoC (%)']
        financial_gain = energy_revenue - energy_cost - degradation_cost
        comparison_data.append({'Strategia': strategy, 'Guadagno Finanziario (€)': financial_gain, 'Costo Virtuale (€)': anxiety_cost, 'SOC Finale (%)': round(soc_final, 1)})
    return pd.DataFrame(comparison_data).sort_values(by='Guadagno Finanziario (€)', ascending=False)

def get_user_input(prompt: str, type_converter, validator, error_message="Input non valido. Riprova."):
    while True:
        try:
            value = type_converter(input(prompt))
            if validator(value): return value
            else: print(error_message)
        except (ValueError, TypeError): print("Formato input non valido. Inserire un numero.")

# ========================================================================
# LOGICA DI ESECUZIONE CENTRALIZZATA
# ========================================================================
def run_comparison_simulation(vehicle_config: Dict, sim_config: Dict, test_day: Dict, agent_types: List[str]):
    try:
        if not validate_config(sim_config, vehicle_config): return
    except ValueError as e:
        print(f"ERRORE DI CONFIGURAZIONE: {e}", file=sys.stderr); return

    chem_name = vehicle_config['chem_name']
    base_profile_name = sim_config.get('profile_name', 'custom') # Usa il nome del profilo se disponibile
    results = {}; optimizer = V2GOptimizer(vehicle_config, sim_config)
    optimizer.set_prices_for_simulation(test_day)
    
    # Esecuzione strategie
    if 'euristica' in agent_types: results['Euristica'] = optimizer.run_heuristic_strategy()
    if 'lcvf' in agent_types: results['LCVF'] = optimizer.run_lcvf_strategy()
    if 'mpc' in agent_types: results[f"MPC (H={sim_config['mpc_horizon']}h)"] = optimizer.run_mpc_strategy()

    if 'q_table' in agent_types and base_profile_name != 'custom':
        q_table_path = os.path.join('q_tables', f"q_table_{base_profile_name}_{chem_name}.npy")
        if os.path.exists(q_table_path):
            q_table = np.load(q_table_path); results[f'Q-Table'] = optimizer.run_rl_strategy(q_table)
        else: print(f"ATTENZIONE: Q-Table '{q_table_path}' non trovata.")

    if 'specialist' in agent_types and TORCH_AVAILABLE and base_profile_name != 'custom':
        dqn_spec_path = f"dqn_model_{base_profile_name}_{chem_name}.pth"
        if os.path.exists(dqn_spec_path):
            agent = DQNAgentForEval(DQN_PARAMS['state_size'], DQN_PARAMS['hidden_size']); agent.load_model(dqn_spec_path)
            results[f'DQN Specialista'] = optimizer.run_dqn_strategy(agent, is_generalist=False)
        else: print(f"ATTENZIONE: Modello DQN Specialista '{dqn_spec_path}' non trovato.")

    if 'generalist' in agent_types and TORCH_AVAILABLE:
        dqn_gen_path = "dqn_model_generalist.pth"
        if os.path.exists(dqn_gen_path):
            agent = DQNAgentForEval(DQN_GENERALIST_PARAMS['state_size'], DQN_GENERALIST_PARAMS['hidden_size']); agent.load_model(dqn_gen_path)
            results['DQN Generalista'] = optimizer.run_dqn_strategy(agent, is_generalist=True)
        else: print(f"ATTENZIONE: Modello DQN Generalista '{dqn_gen_path}' non trovato.")
    
    # Stampa risultati
    print("\n" + "="*80)
    print(f"RISULTATI CONFRONTO - Batteria: {chem_name.upper()}, Profilo: {base_profile_name.capitalize()}")
    print(f"Parametri: SoC Init={sim_config['initial_soc']*100:.0f}%, Min={sim_config['soc_min_utente']*100:.0f}%, Target={sim_config['soc_target_finale']*100:.0f}%")
    print("="*80)
    
    if not results: print("Nessuna strategia è stata eseguita."); return
    for name, df in results.items():
        print(f"\n--- Dettaglio Orario Strategia: {name} ---")
        df_display = df.copy()
        for col in ['SoC Iniziale (%)', 'Variazione SoC (%)']: df_display[col] = df_display[col].map('{:,.1f}'.format)
        print(df_display[['Ora', 'Azione', 'SoC Iniziale (%)', 'Variazione SoC (%)', 'Costo Energia (€)', 'Ricavo Energia (€)', 'Costo Degradazione (€)', 'Costo Ansia (€)']].to_string(index=False))
        
    print("\n" + "="*80); print("RIEPILOGO FINANZIARIO E PRESTAZIONALE"); print("="*80)
    print(compare_strategies(results).to_string(index=False))
    print("="*80)
    
def get_custom_sim_config():
    min_batt, max_batt = BASE_VEHICLE_PARAMS['soc_min_batteria'], BASE_VEHICLE_PARAMS['soc_max']
    initial_soc = get_user_input(f"SoC Iniziale ({min_batt*100:.0f}-{max_batt*100:.0f}%): ", float, lambda x: min_batt <= x/100 <= max_batt, "Valore non valido.") / 100
    soc_min_utente = get_user_input(f"SoC Minimo Utente (>{min_batt*100:.0f}% e <{max_batt*100:.0f}%): ", float, lambda x: min_batt < x/100 < max_batt, "Valore non valido.") / 100
    soc_target_finale = get_user_input(f"SoC Target Finale (>= {min_batt*100:.0f}% e <= {max_batt*100:.0f}%): ", float, lambda x: min_batt <= x/100 <= max_batt, "Valore non valido.") / 100
    penalita_ansia = USER_PROFILES['bilanciato']['penalita_ansia']
    return {**BASE_SIMULATION_PARAMS, 'initial_soc': initial_soc, 'soc_min_utente': soc_min_utente, 'soc_target_finale': soc_target_finale, 'penalita_ansia': penalita_ansia}

def main():
    price_data = load_price_data()
    test_profiles = create_test_profiles(price_data)
    if not test_profiles: print("Nessun profilo di prezzo di test disponibile. Uscita.", file=sys.stderr); return

    while True:
        print("\n" + "--- MENU PRINCIPALE - CRUSCOTTO DI ANALISI V2G (V4.0) ---")
        print("1) Confronto Agenti SPECIALISTI (Q-Table, DQN) vs Classici")
        print("2) Confronto Agente GENERALISTA vs Strategie Adattive (MPC, Euristiche)")
        print("3) Analisi di Sensibilità su CHIMICA (per Specialisti)")
        print("4) Analisi di Sensibilità su SOC MINIMO UTENTE (per Generalista e Adattive)")
        print("5) Pulisci tutti i modelli e le Q-table addestrate")
        print("6) Esci")
        choice = input("Scelta [1-6]: ").strip()

        test_day_for_session = random.choice(test_profiles)
        print(f"\nINFO: Tutte le simulazioni in questa sessione useranno lo stesso giorno di test casuale.")
        
        profile_map = {str(i+1): p for i, p in enumerate(USER_PROFILES.keys())}
        chem_map = {str(i+1): c for i, c in enumerate(BATTERY_CHEMISTRIES.keys())}
            
        if choice == '1':
            print("\n--- Confronto Agenti Specialisti ---")
            p_choice = get_user_input("Scegli un profilo utente [1-Cons, 2-Bil, 3-Aggr]: ", str, lambda x: x in profile_map); profile_name = profile_map[p_choice]
            c_choice = get_user_input("Scegli una chimica [1-NCA, 2-LFP, 3-Semp]: ", str, lambda x: x in chem_map); chem_name = chem_map[c_choice]
            sim_config = {**BASE_SIMULATION_PARAMS, **USER_PROFILES[profile_name], 'profile_name': profile_name}
            vehicle_config = {**BASE_VEHICLE_PARAMS, **BATTERY_CHEMISTRIES[chem_name], 'chem_name': chem_name}
            run_comparison_simulation(vehicle_config, sim_config, test_day_for_session, ['euristica', 'lcvf', 'mpc', 'q_table', 'specialist'])

        elif choice == '2':
            print("\n--- Confronto Agente Generalista ---")
            sim_config = get_custom_sim_config()
            c_choice = get_user_input("Scegli una chimica [1-NCA, 2-LFP, 3-Semp]: ", str, lambda x: x in chem_map); chem_name = chem_map[c_choice]
            vehicle_config = {**BASE_VEHICLE_PARAMS, **BATTERY_CHEMISTRIES[chem_name], 'chem_name': chem_name}
            run_comparison_simulation(vehicle_config, sim_config, test_day_for_session, ['euristica', 'lcvf', 'mpc', 'generalist'])

        elif choice == '3':
            print("\n--- Analisi di Sensibilità sulla Chimica (per Specialisti) ---")
            p_choice = get_user_input("Scegli un profilo utente fisso [1-Cons, 2-Bil, 3-Aggr]: ", str, lambda x: x in profile_map); profile_name = profile_map[p_choice]
            sim_config = {**BASE_SIMULATION_PARAMS, **USER_PROFILES[profile_name], 'profile_name': profile_name}
            for chem_name in BATTERY_CHEMISTRIES.keys():
                vehicle_config = {**BASE_VEHICLE_PARAMS, **BATTERY_CHEMISTRIES[chem_name], 'chem_name': chem_name}
                run_comparison_simulation(vehicle_config, sim_config, test_day_for_session, ['euristica', 'lcvf', 'mpc', 'q_table', 'specialist'])

        elif choice == '4':
            print("\n--- Analisi di Sensibilità su SoC Minimo Utente (per Agenti Adattivi) ---")
            min_batt, max_batt = BASE_VEHICLE_PARAMS['soc_min_batteria'], BASE_VEHICLE_PARAMS['soc_max']
            print("Inserisci i parametri FISSI per entrambe le simulazioni:")
            initial_soc = get_user_input(f"SoC Iniziale ({min_batt*100:.0f}-{max_batt*100:.0f}%): ", float, lambda x: min_batt <= x/100 <= max_batt, "Valore non valido.") / 100
            soc_target_finale = get_user_input(f"SoC Target Finale ({min_batt*100:.0f}-{max_batt*100:.0f}%): ", float, lambda x: min_batt <= x/100 <= max_batt, "Valore non valido.") / 100
            c_choice = get_user_input("Scegli una chimica [1-NCA, 2-LFP, 3-Semp]: ", str, lambda x: x in chem_map); chem_name = chem_map[c_choice]
            vehicle_config = {**BASE_VEHICLE_PARAMS, **BATTERY_CHEMISTRIES[chem_name], 'chem_name': chem_name}

            print("\nOra inserisci i due valori di SoC Minimo Utente da confrontare:")
            soc_min_validator = lambda x: min_batt < x/100 < max_batt
            error_msg = f"Valore non valido. Deve essere > {min_batt*100:.0f} e < {max_batt*100:.0f}."
            soc_min_1 = get_user_input(f"PRIMO SoC Minimo Utente (%): ", float, soc_min_validator, error_msg) / 100
            soc_min_2 = get_user_input(f"SECONDO SoC Minimo Utente (%): ", float, soc_min_validator, error_msg) / 100
            
            penalita_ansia = USER_PROFILES['bilanciato']['penalita_ansia']
            
            print("\n" + "#"*35 + " ESEGUO SIMULAZIONE 1 " + "#"*32)
            sim_config_1 = {**BASE_SIMULATION_PARAMS, 'initial_soc': initial_soc, 'soc_min_utente': soc_min_1, 'soc_target_finale': soc_target_finale, 'penalita_ansia': penalita_ansia}
            run_comparison_simulation(vehicle_config, sim_config_1, test_day_for_session, ['euristica', 'lcvf', 'mpc', 'generalist'])

            print("\n" + "#"*35 + " ESEGUO SIMULAZIONE 2 " + "#"*32)
            sim_config_2 = {**BASE_SIMULATION_PARAMS, 'initial_soc': initial_soc, 'soc_min_utente': soc_min_2, 'soc_target_finale': soc_target_finale, 'penalita_ansia': penalita_ansia}
            run_comparison_simulation(vehicle_config, sim_config_2, test_day_for_session, ['euristica', 'lcvf', 'mpc', 'generalist'])

        elif choice == '5':
            print("\n--- Pulizia dei Modelli Addestrati ---")
            confirm = input("Sei sicuro di voler eliminare tutti i file .pth e .npy? [s/N]: ").strip().lower()
            if confirm == 's':
                deleted_files = 0; q_tables_dir = 'q_tables'
                for dirpath, _, filenames in os.walk('.'):
                    if 'q_tables' in dirpath: continue
                    for f in filenames:
                        if f.endswith(('.pth', '.npy')):
                            os.remove(os.path.join(dirpath, f)); deleted_files += 1; print(f"Eliminato: {os.path.join(dirpath, f)}")
                if os.path.exists(q_tables_dir):
                    for f in os.listdir(q_tables_dir):
                        if f.endswith('.npy'):
                            file_path = os.path.join(q_tables_dir, f); os.remove(file_path); deleted_files += 1; print(f"Eliminato: {file_path}")
                print(f"Pulizia completata. {deleted_files} file eliminati.")
            else: print("Azione annullata.")

        elif choice == '6': print("Uscita."); break
        else: print("Scelta non valida.")

if __name__ == "__main__":
    main()
