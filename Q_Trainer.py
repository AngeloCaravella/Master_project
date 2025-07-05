# ========================================================================
# V2G Q-TABLE BATCH TRAINER
# Author: Angelo Caravella 
# Version: 1.2
# Description: Aggiunge una logica di Early Stopping robusta basata su una
#              soglia di miglioramento minima (min_delta).
# ========================================================================

import numpy as np
import pandas as pd
import random
import os
import sys
from typing import Dict, List, Tuple
import multiprocessing

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
        'degradation_model': 'nca',
        'costo_batteria': 120  # €/kWh - più costosa, alta densità energetica
    },
    'lfp': {
        'degradation_model': 'lfp',
        'costo_batteria': 90   # €/kWh - più economica, meno densa, più cicli
    },
    'semplice': {
        'degradation_model': 'simple',
        'costo_batteria': 70   # €/kWh - modello iper-semplificato o legacy
    }
}

BASE_VEHICLE_PARAMS = {
    'capacita': 60, 'p_carica': 7.4, 'p_scarica': 5.0, 'efficienza_carica': 0.95,
    'efficienza_scarica': 0.95, 'soc_max': 0.9, 'soc_min_batteria': 0.1, 'lfp_k_slope': 0.0035,
}

RL_PARAMS = {
    'states_ora': 24, 'states_soc': 11, 'states_battery': 3, 'alpha': 0.1, 'gamma': 0.98,
    'epsilon': 0.01, 'episodes': 100000,
    'early_stopping_patience': 100, # N. di controlli (x100 episodi)
    'early_stopping_min_delta': 0.01 # Miglioramento minimo per resettare la pazienza
}

# ========================================================================
# CLASSI DI SIMULAZIONE (Invariate)
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
        # Mappatura da nome chimica a indice numerico
        battery_chem_map = {name: i for i, name in enumerate(BATTERY_CHEMISTRIES.keys())}
        battery_idx = battery_chem_map[self.degradation_model_type]

        # La Q-table ora ha una dimensione in più per la chimica della batteria
        q_table = np.zeros((rl_p['states_ora'], rl_p['states_soc'], rl_p['states_battery'], len(self.actions_map)))
        best_q_table = np.copy(q_table)
        epsilon = rl_p['epsilon']
        
        best_avg_reward = -np.inf
        patience = rl_p['early_stopping_patience']
        min_delta = rl_p['early_stopping_min_delta']
        patience_counter = 0
        total_rewards = []

        for episode in range(rl_p['episodes']):
            episode_prices = random.choice(training_daily_profiles)
            soc = self.sim_params['initial_soc']
            episode_reward = 0
            for ora in range(rl_p['states_ora']):
                soc_discrete = self._discretize_soc(soc, rl_p['states_soc'])
                # Indicizzazione con la nuova dimensione
                azione = random.randint(0, len(self.actions_map) - 1) if random.random() < epsilon else np.argmax(q_table[ora, soc_discrete, battery_idx])
                new_soc, reward = self._get_rl_step_result(soc, ora, azione, episode_prices, last_hour=(ora == rl_p['states_ora'] - 1))
                episode_reward += reward
                new_soc_discrete = self._discretize_soc(new_soc, rl_p['states_soc'])
                next_q_value = 0
                if ora < rl_p['states_ora'] - 1: 
                    next_q_value = np.max(q_table[ora + 1, new_soc_discrete, battery_idx])
                current_q = q_table[ora, soc_discrete, battery_idx, azione]
                new_q = (1 - rl_p['alpha']) * current_q + rl_p['alpha'] * (reward + rl_p['gamma'] * next_q_value)
                q_table[ora, soc_discrete, battery_idx, azione] = new_q
                soc = new_soc
            
            total_rewards.append(episode_reward)

            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(total_rewards[-100:])
                print(f"Episodio {episode + 1}/{rl_p['episodes']}, Avg Reward (100 ep): {avg_reward:.2f}, Epsilon: {epsilon:.4f}")
                if avg_reward > best_avg_reward + min_delta:
                    best_avg_reward = avg_reward
                    best_q_table = np.copy(q_table)
                    patience_counter = 0
                    print(f"  -> Nuovo record di Avg Reward! Pazienza resettata.")
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"\nATTENZIONE: Nessun miglioramento significativo per {patience * 100} episodi. Interruzione anticipata.")
                    break

        q_table_dir = os.path.dirname(q_table_path)
        if q_table_dir and not os.path.exists(q_table_dir): os.makedirs(q_table_dir)
        np.save(q_table_path, best_q_table)
        print(f"--- Addestramento completato! Tabella migliore salvata in '{q_table_path}' con Avg Reward: {best_avg_reward:.2f} ---")

# ========================================================================
# FUNZIONI AUSILIARIE (Invariate)
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
# ESECUZIONE AUTOMATICA (Invariata)
# ========================================================================

def train_single_q_table(args):
    profile_name, chem_name, training_profiles, BASE_VEHICLE_PARAMS, USER_PROFILES, BATTERY_CHEMISTRIES = args

    q_table_path = os.path.join('q_tables', f"q_table_{profile_name}_{chem_name}.npy")
    print("\n" + "-"*60)
    print(f"Verifica combinazione: Profilo '{profile_name.capitalize()}', Batteria '{chem_name.upper()}'")
    print(f"File tabella target: '{q_table_path}'")

    if os.path.exists(q_table_path):
        print("STATO: Trovata. Addestramento saltato.")
        return 0 # 0 trained, 1 skipped
    
    print("STATO: Non trovata. Avvio addestramento...")

    sim_config = {**USER_PROFILES[profile_name]}
    vehicle_config = {**BASE_VEHICLE_PARAMS, **BATTERY_CHEMISTRIES[chem_name]}
    
    trainer = RLAgentTrainer(vehicle_config, sim_config)
    trainer.train(training_profiles, q_table_path)
    return 1 # 1 trained, 0 skipped

def main():
    print("--- TRAINER AUTOMATICO PER Q-TABLES V2G ---")
    print("Controllo e addestramento di tutte le tabelle mancanti...")

    price_data = load_price_data()
    training_profiles = create_daily_profiles(price_data)
    
    if not training_profiles: 
        print("Nessun profilo di prezzo valido per il training. Uscita.", file=sys.stderr)
        return

    tasks = []
    for profile_name, _ in USER_PROFILES.items():
        for chem_name, _ in BATTERY_CHEMISTRIES.items():
            tasks.append((profile_name, chem_name, training_profiles, BASE_VEHICLE_PARAMS, USER_PROFILES, BATTERY_CHEMISTRIES))

    trained_count, skipped_count = 0, 0

    # Usa un Pool di processi per eseguire gli addestramenti in parallelo
    # Il numero di processi è di default il numero di core della CPU
    with multiprocessing.Pool() as pool:
        results = pool.map(train_single_q_table, tasks)
    
    trained_count = sum(results)
    skipped_count = len(results) - trained_count

    print("\n" + "="*60)
    print("PROCESSO DI ADDESTRAMENTO DI MASSA COMPLETATO")
    print(f"  - Tabelle addestrate in questa sessione: {trained_count}")
    print(f"  - Tabelle già esistenti e saltate: {skipped_count}")
    print(f"  - Totale tabelle verificate: {trained_count + skipped_count}")
    print("="*60)

if __name__ == "__main__":
    # Questo è fondamentale per il multiprocessing su Windows
    multiprocessing.freeze_support()
    main()
