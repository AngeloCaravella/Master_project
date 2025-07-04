

# ========================================================================
# V2G Q-TABLE BATCH TRAINER
# Author: Angelo Caravella & Gemini
# Version: 1.0
# Description: Strumento di addestramento automatico e di massa per Q-table.
#              Lo script controlla tutte le tabelle mancanti per ogni combinazione
#              di profilo/batteria e le addestra in autonomia.
# ========================================================================

import numpy as np
import pandas as pd
import random
import os
import sys
from typing import Dict, List, Tuple

# ========================================================================
# CONFIGURAZIONI PREDEFINITE (Copiate da New.py per coerenza)
# ========================================================================

USER_PROFILES = {
    'conservativo': {
        'soc_min_utente': 0.60, 'penalita_ansia': 0.02, 'soc_target_finale': 0.70,
    },
    'bilanciato': {
        'soc_min_utente': 0.30, 'penalita_ansia': 0.01, 'soc_target_finale': 0.50,
    },
    'aggressivo': {
        'soc_min_utente': 0.15, 'penalita_ansia': 0.005, 'soc_target_finale': 0.20,
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
    'initial_soc': 0.5,
}

RL_PARAMS = {
    'states_ora': 24, 'states_soc': 11, 'alpha': 0.1, 'gamma': 0.98,
    'epsilon': 1.0, 'epsilon_decay': 0.99985, 'epsilon_min': 0.01, 'episodes': 100000,
}

# ========================================================================
# CLASSI DI SIMULAZIONE (Minime, per il training)
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

class RLAgentTrainer:
    def __init__(self, vehicle_config: Dict, sim_config: Dict):
        self.vehicle_params = vehicle_config
        self.sim_params = sim_config
        self.actions_map = {0: 'Attesa', 1: 'Carica', 2: 'Scarica'}
        self.degradation_model_type = self.vehicle_params.get('degradation_model', 'simple')
        self.degradation_calculator = BatteryDegradationModel(self.vehicle_params)

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

    def _get_rl_step_result(self, soc: float, ora: int, azione: int, prices: Dict[int, float], last_hour: bool) -> Tuple[float, float]:
        new_soc, reward, energy_processed_kwh = soc, 0.0, 0
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
        if last_hour: reward -= self._calculate_terminal_soc_cost(new_soc)
        return np.clip(new_soc, 0, 1), reward

    def _discretize_soc(self, soc: float, states: int) -> int:
        return int(np.clip(round(soc * (states - 1)), 0, states - 1))

    def train(self, training_daily_profiles: List[Dict[int, float]], q_table_path: str):
        print(f"\n--- AVVIO ADDESTRAMENTO PER Q-TABLE '{q_table_path}' ---")
        rl_p = RL_PARAMS
        q_table = np.zeros((rl_p['states_ora'], rl_p['states_soc'], len(self.actions_map)))
        epsilon = rl_p['epsilon']
        for episode in range(rl_p['episodes']):
            episode_prices = random.choice(training_daily_profiles)
            soc = self.sim_params['initial_soc']
            for ora in range(rl_p['states_ora']):
                soc_discrete = self._discretize_soc(soc, rl_p['states_soc'])
                azione = random.randint(0, len(self.actions_map) - 1) if random.random() < epsilon else np.argmax(q_table[ora, soc_discrete])
                new_soc, reward = self._get_rl_step_result(soc, ora, azione, episode_prices, last_hour=(ora == rl_p['states_ora'] - 1))
                new_soc_discrete = self._discretize_soc(new_soc, rl_p['states_soc'])
                next_q_value = 0
                if ora < rl_p['states_ora'] - 1: next_q_value = np.max(q_table[ora + 1, new_soc_discrete])
                current_q = q_table[ora, soc_discrete, azione]
                new_q = (1 - rl_p['alpha']) * current_q + rl_p['alpha'] * (reward + rl_p['gamma'] * next_q_value)
                q_table[ora, soc_discrete, azione] = new_q
                soc = new_soc
            epsilon = max(rl_p['epsilon_min'], epsilon * rl_p['epsilon_decay'])
            if (episode + 1) % 20000 == 0: print(f"Episodio {episode + 1}/{rl_p['episodes']}, Epsilon: {epsilon:.4f}")
        q_table_dir = os.path.dirname(q_table_path)
        if q_table_dir and not os.path.exists(q_table_dir): os.makedirs(q_table_dir)
        np.save(q_table_path, q_table)
        print(f"--- Addestramento per '{q_table_path}' completato! ---")

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

def create_daily_profiles(df: pd.DataFrame, test_zone: str = "Italia") -> List[Dict]:
    all_zones = [col for col in df.columns if col not in ['Ora', 'Data']]
    training_zones = [z for z in all_zones if z != test_zone]
    training_profiles = []
    for zone in training_zones:
        zone_prices = df[zone].dropna()
        for i in range(0, len(zone_prices), 24):
            if len(zone_prices.iloc[i:i+24]) == 24: training_profiles.append({h: p for h, p in enumerate(zone_prices.iloc[i:i+24])})
    return training_profiles

# ========================================================================
# ESECUZIONE AUTOMATICA
# ========================================================================

def main():
    print("--- TRAINER AUTOMATICO PER Q-TABLES V2G ---")
    print("Controllo e addestramento di tutte le tabelle mancanti...")

    price_data = load_price_data()
    training_profiles = create_daily_profiles(price_data)
    
    if not training_profiles: 
        print("Nessun profilo di prezzo valido per il training. Uscita.", file=sys.stderr)
        return

    trained_count, skipped_count = 0, 0

    for profile_name, profile_params in USER_PROFILES.items():
        for chem_name, chem_params in BATTERY_CHEMISTRIES.items():
            q_table_path = os.path.join('q_tables', f"q_table_{profile_name}_{chem_name}.npy")
            print("\n" + "-"*60)
            print(f"Verifica combinazione: Profilo '{profile_name.capitalize()}', Batteria '{chem_name.upper()}'")
            print(f"File tabella target: '{q_table_path}'")

            if os.path.exists(q_table_path):
                print("STATO: Trovata. Addestramento saltato.")
                skipped_count += 1
                continue
            
            print("STATO: Non trovata. Avvio addestramento...")
            trained_count += 1

            sim_config = {**BASE_SIMULATION_PARAMS, **profile_params}
            vehicle_config = {**BASE_VEHICLE_PARAMS, **chem_params}
            
            trainer = RLAgentTrainer(vehicle_config, sim_config)
            trainer.train(training_profiles, q_table_path)

    print("\n" + "="*60)
    print("PROCESSO DI ADDESTRAMENTO DI MASSA COMPLETATO")
    print(f"  - Tabelle addestrate in questa sessione: {trained_count}")
    print(f"  - Tabelle già esistenti e saltate: {skipped_count}")
    print(f"  - Totale tabelle verificate: {trained_count + skipped_count}")
    print("="*60)

if __name__ == "__main__":
    main()
