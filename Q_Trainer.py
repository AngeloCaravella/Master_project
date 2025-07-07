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
    'states_ora': 24, 'states_soc': 11, 'alpha': 0.1, 'gamma': 0.98,
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

class V2G_Environment:
    def __init__(self, daily_price_profiles: List[Dict[int, float]]):
        self.daily_profiles = daily_price_profiles
        self.actions_map = {0: 'Attesa', 1: 'Carica', 2: 'Scarica'}
        self.max_price_for_norm = 0.5  # Non usato qui, ma per coerenza
        self.vehicle_params, self.sim_params, self.degradation_calc = None, None, None

    def reset(self, dynamic_vehicle_config: Dict, dynamic_sim_params: Dict) -> None:
        self.current_hour = 0
        self.vehicle_params = dynamic_vehicle_config
        self.sim_params = dynamic_sim_params
        self.degradation_model_type = self.vehicle_params['degradation_model']
        self.degradation_calc = BatteryDegradationModel(self.vehicle_params)
        self.current_soc = self.sim_params['initial_soc']
        self.current_prices = random.choice(self.daily_profiles)

    def _calculate_degradation_cost(self, soc_start: float, soc_end: float, energy_kwh: float) -> float:
        if self.degradation_model_type == 'lfp':
            return self.degradation_calc.cost_lfp_model(energy_kwh)
        elif self.degradation_model_type == 'nca':
            return self.degradation_calc.cost_nca_model(soc_start, soc_end)
        else:
            return self.degradation_calc.cost_simple_linear(energy_kwh)

    def _calculate_reward(self, revenue, cost, degradation, soc_end, action, soc_start) -> float:
        reward = revenue - cost - degradation
        if soc_end < self.sim_params['soc_min_utente']:
            reward -= self.sim_params['penalita_ansia'] * (self.sim_params['soc_min_utente'] - soc_end) * 100
        
        action_str = self.actions_map[action]
        if (action_str == 'Carica' and soc_start >= self.vehicle_params['soc_max']) or \
           (action_str == 'Scarica' and soc_start <= self.vehicle_params['soc_min_batteria']):
            reward -= 0.1
        
        done = self.current_hour >= 23
        if done:
            target_soc = self.sim_params['soc_target_finale']
            if soc_end < target_soc:
                reward -= self.sim_params['penalita_ansia'] * 20 * (target_soc - soc_end) * 100
            else:
                reward += 5.0
        return reward

    def step(self, action: int) -> Tuple[float, float, bool]:
        soc_start = self.current_soc
        price = self.current_prices.get(self.current_hour, 0)
        power_kwh = 0
        
        if self.actions_map[action] == 'Carica' and soc_start < self.vehicle_params['soc_max']:
            power_kwh = self.vehicle_params['p_carica']
        elif self.actions_map[action] == 'Scarica' and soc_start > self.vehicle_params['soc_min_batteria']:
            power_kwh = -self.vehicle_params['p_scarica']
            
        energy_cost, energy_revenue, energy_processed_kwh = 0, 0, 0
        soc_end = soc_start

        if power_kwh > 0:
            energy_stored = power_kwh * self.vehicle_params['efficienza_carica']
            soc_end += energy_stored / self.vehicle_params['capacita']
            energy_cost = price * power_kwh
            energy_processed_kwh = energy_stored
        elif power_kwh < 0:
            energy_drawn = -power_kwh / self.vehicle_params['efficienza_scarica']
            soc_end -= energy_drawn / self.vehicle_params['capacita']
            energy_revenue = price * -power_kwh
            energy_processed_kwh = -energy_drawn
            
        soc_end = np.clip(soc_end, 0, 1)
        degradation_cost = self._calculate_degradation_cost(soc_start, soc_end, energy_processed_kwh)
        
        reward = self._calculate_reward(energy_revenue, energy_cost, degradation_cost, soc_end, action, soc_start)
        
        self.current_soc = soc_end
        self.current_hour += 1
        done = self.current_hour >= 24
        
        return self.current_soc, reward, done

def worker_process(worker_id, env_config_queue, episode_result_queue, daily_price_profiles, q_table_flat_shared, q_table_shape):
    # Funzione eseguita da ogni processo worker.
    # Esegue un episodio e restituisce il reward totale.
    env = V2G_Environment(daily_price_profiles)
    q_table = np.frombuffer(q_table_flat_shared).reshape(q_table_shape)
    
    while True:
        msg = env_config_queue.get()
        if msg == "STOP":
            break
            
        vehicle_config, sim_config, epsilon = msg

        env.reset(vehicle_config, sim_config)
        
        episode_reward = 0
        done = False
        
        while not done:
            soc_discrete = int(np.clip(round(env.current_soc * (RL_PARAMS['states_soc'] - 1)), 0, RL_PARAMS['states_soc'] - 1))
            
            if random.random() < epsilon:
                action = random.randint(0, len(env.actions_map) - 1)
            else:
                action = np.argmax(q_table[env.current_hour, soc_discrete])

            _, reward, done = env.step(action)
            episode_reward += reward
            
        episode_result_queue.put(episode_reward)

class RLAgentTrainer:
    def __init__(self, vehicle_config: Dict, sim_config: Dict):
        self.vehicle_params = vehicle_config
        self.sim_params = sim_config
        self.actions_map = {0: 'Attesa', 1: 'Carica', 2: 'Scarica'}

    def _discretize_soc(self, soc: float, states: int) -> int:
        return int(np.clip(round(soc * (states - 1)), 0, states - 1))

    def train(self, training_daily_profiles: List[Dict[int, float]], q_table_path: str):
        print(f"\n--- AVVIO ADDESTRAMENTO PER Q-TABLE '{q_table_path}' ---")
        rl_p = RL_PARAMS
        
        if os.path.exists(q_table_path):
            print(f"STATO: Trovata Q-table esistente '{q_table_path}'.")
            q_table = np.load(q_table_path)
        else:
            q_table = np.zeros((rl_p['states_ora'], rl_p['states_soc'], len(self.actions_map)))

        best_q_table = np.copy(q_table)
        epsilon = rl_p['epsilon']
        
        best_avg_reward = -np.inf
        patience = rl_p['early_stopping_patience']
        min_delta = rl_p['early_stopping_min_delta']
        patience_counter = 0
        total_rewards = []

        num_workers = multiprocessing.cpu_count()
        env_config_queue = multiprocessing.Queue()
        episode_result_queue = multiprocessing.Queue()
        
        # Creazione di un buffer di memoria condivisa per la Q-table
        q_table_shape = q_table.shape
        q_table_flat = multiprocessing.RawArray('d', q_table.flatten())
        
        workers = []
        for i in range(num_workers):
            p = multiprocessing.Process(target=worker_process, args=(i, env_config_queue, episode_result_queue, training_daily_profiles, q_table_flat, q_table_shape))
            workers.append(p)
            p.start()

        try:
            for episode in range(rl_p['episodes']):
                # Aggiorna la Q-table condivisa con la versione più recente
                q_table_shared = np.frombuffer(q_table_flat).reshape(q_table_shape)
                np.copyto(q_table_shared, q_table)

                # Invia le configurazioni ai worker
                for _ in range(num_workers):
                    env_config_queue.put((self.vehicle_params, self.sim_params, epsilon))

                # Raccogli i risultati e aggiorna la Q-table
                for _ in range(num_workers):
                    episode_reward = episode_result_queue.get()
                    total_rewards.append(episode_reward)
                
                # L'aggiornamento della Q-table avviene nel processo principale (Learner)
                # Per semplicità, questo esempio parallelizza la raccolta di esperienze (rewards)
                # e l'aggiornamento della Q-table rimane seriale.
                # Si potrebbe parallelizzare anche l'aggiornamento, ma aumenterebbe la complessità.
                
                # Esegui un episodio di training per aggiornare la Q-table
                episode_prices = random.choice(training_daily_profiles)
                soc = self.sim_params['initial_soc']
                for ora in range(rl_p['states_ora']):
                    soc_discrete = self._discretize_soc(soc, rl_p['states_soc'])
                    azione = random.randint(0, len(self.actions_map) - 1) if random.random() < epsilon else np.argmax(q_table[ora, soc_discrete])
                    
                    # Usiamo un ambiente locale per ottenere il reward per l'aggiornamento
                    env = V2G_Environment(training_daily_profiles)
                    env.reset(self.vehicle_params, self.sim_params)
                    env.current_hour = ora
                    env.current_soc = soc
                    
                    new_soc, reward, _ = env.step(azione)

                    new_soc_discrete = self._discretize_soc(new_soc, rl_p['states_soc'])
                    next_q_value = 0
                    if ora < rl_p['states_ora'] - 1:
                        next_q_value = np.max(q_table[ora + 1, new_soc_discrete])
                    
                    current_q = q_table[ora, soc_discrete, azione]
                    new_q = (1 - rl_p['alpha']) * current_q + rl_p['alpha'] * (reward + rl_p['gamma'] * next_q_value)
                    q_table[ora, soc_discrete, azione] = new_q
                    soc = new_soc

                if (episode + 1) % 100 == 0 and len(total_rewards) >= 100:
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
        finally:
            for _ in range(num_workers):
                env_config_queue.put("STOP")
            for p in workers:
                p.join()

        q_table_dir = os.path.dirname(q_table_path)
        if q_table_dir and not os.path.exists(q_table_dir):
            os.makedirs(q_table_dir)
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

def create_daily_profiles(df: pd.DataFrame, target_zone: str, num_days: int) -> List[Dict]:
    if target_zone not in df.columns: print(f"ERRORE: La colonna '{target_zone}' non esiste.", file=sys.stderr); sys.exit(1)
    print(f"INFO: Estraggo dati dalla colonna '{target_zone}'.")
    all_profiles_from_zone = []
    zone_prices = df[target_zone].dropna()
    for i in range(0, len(zone_prices), 24):
        if len(zone_prices.iloc[i:i+24]) == 24: all_profiles_from_zone.append({h: p for h, p in enumerate(zone_prices.iloc[i:i+24])})
    if not all_profiles_from_zone: print(f"ERRORE: Nessun profilo completo trovato per '{target_zone}'.", file=sys.stderr); sys.exit(1)
    random.shuffle(all_profiles_from_zone)
    selected_profiles = all_profiles_from_zone[:min(num_days, len(all_profiles_from_zone))]
    print(f"INFO: Trovati {len(all_profiles_from_zone)} giorni. Selezionati {len(selected_profiles)} giorni per il training.")
    return selected_profiles

# ========================================================================
# ESECUZIONE AUTOMATICA (Invariata)
# ========================================================================

def get_user_input(prompt: str, type_converter, validator, error_message="Input non valido. Riprova."):
    while True:
        try:
            value = type_converter(input(prompt))
            if validator(value):
                return value
            else:
                print(error_message)
        except (ValueError, TypeError):
            print("Formato input non valido. Inserire un numero.")

def main():
    print("--- TRAINER AUTOMATICO PER Q-TABLES V2G ---")
    print("Controllo e addestramento di tutte le tabelle mancanti...")

    price_data = load_price_data()
    available_zones = [col for col in price_data.columns if col not in ['Ora', 'Data']]
    
    print("Zone di prezzo disponibili:")
    for i, zone in enumerate(available_zones):
        print(f"  {i+1}) {zone}")
    
    zone_choice_idx = get_user_input(f"Scegli la zona di prezzo [1-{len(available_zones)}]: ", int, lambda x: 1 <= x <= len(available_zones)) - 1
    target_zone = available_zones[zone_choice_idx]

    max_days = len(price_data) // 24
    num_days = get_user_input(f"Quanti giorni di dati usare per il training? (min 1, max {max_days}): ", int, lambda x: 1 <= x <= max_days)

    training_profiles = create_daily_profiles(price_data, target_zone=target_zone, num_days=num_days)
    
    if not training_profiles: 
        print("Nessun profilo di prezzo valido per il training. Uscita.", file=sys.stderr)
        return

    trained_count, skipped_count = 0, 0

    for profile_name, profile_params in USER_PROFILES.items():
        for chem_name, chem_params in BATTERY_CHEMISTRIES.items():
            q_table_path = os.path.join('q_tables', f"q_table_{profile_name}_{chem_name}.npy")
            print("" + "-"*60)
            print(f"Verifica combinazione: Profilo '{profile_name.capitalize()}', Batteria '{chem_name.upper()}'")
            print(f"File tabella target: '{q_table_path}'")

            if os.path.exists(q_table_path):
                print(f"STATO: Trovata Q-table esistente '{q_table_path}'.")
                print("1) Salta addestramento")
                print("2) Continua addestramento (Fine-Tune)")
                print("3) Ricomincia da zero (Sovrascrivi)")
                choice = get_user_input("Scegli un'opzione per questa tabella [1-3]: ", str, lambda x: x in ['1', '2', '3'])
                
                if choice == '1':
                    print("Addestramento saltato.")
                    skipped_count += 1
                    continue
                elif choice == '2':
                    print("La Q-table esistente sarà caricata per il fine-tuning.")
                elif choice == '3':
                    print("La Q-table esistente sarà sovrascritta al termine.")
                    os.remove(q_table_path) # Rimuovi per ricominciare da zero
            else:
                print("STATO: Non trovata. Avvio addestramento...")

            trained_count += 1

            sim_config = {**USER_PROFILES[profile_name]}
            vehicle_config = {**BASE_VEHICLE_PARAMS, **BATTERY_CHEMISTRIES[chem_name]}
            
            trainer = RLAgentTrainer(vehicle_config, sim_config)
            trainer.train(training_profiles, q_table_path)

    print("" + "="*60)
    print("PROCESSO DI ADDESTRAMENTO DI MASSA COMPLETATO")
    print(f"  - Tabelle addestrate in questa sessione: {trained_count}")
    print(f"  - Tabelle già esistenti e saltate: {skipped_count}")
    print(f"  - Totale tabelle verificate: {trained_count + skipped_count}")
    print("="*60)

if __name__ == "__main__":
    # Questo è fondamentale per il multiprocessing su Windows
    multiprocessing.freeze_support()
    main()
