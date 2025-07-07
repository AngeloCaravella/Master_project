# =======================================================================
# V2G DQN BATCH TRAINER
# Author: Angelo Caravella 
# Version: 1.5
# Description: Strumento di addestramento automatico per agenti DQN. Implementa
#              profili utente corretti (con SoC iniziale dinamico) e una
#              logica di Early Stopping per un training efficiente.
#
# Requisiti:
# pip install torch pandas numpy openpyxl
# ========================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import os
import sys
from collections import deque
from typing import Dict, List, Tuple
import multiprocessing # Aggiunto l'import di multiprocessing

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

DQN_PARAMS = {
    'state_size': 3, 'action_size': 3, 'hidden_size': 64, 'buffer_size': 100000,
    'batch_size': 64, 'gamma': 0.98, 'learning_rate': 0.001, 'epsilon_start': 0.01,
    'target_update_freq': 10,
    'episodes': 20000, # Riduciamo un po' il massimo, tanto c'è l'early stopping
    'early_stopping_patience': 100, # N. di controlli (x100 episodi)
    'early_stopping_min_delta': 0.01, # Miglioramento minimo per resettare la pazienza
    'worker_batch_size': 64, # Nuovo parametro per il batching delle esperienze dai worker
    'reward_scale': 0.01, # Aggiunto reward_scale per coerenza con il generalista
    'num_workers': 8 # Numero di processi worker
}

# ========================================================================
# CLASSI (Invariate, ma l'ambiente riceve le nuove config)
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
        self.max_price_for_norm = 0.5
        # Questi verranno impostati dinamicamente ad ogni reset
        self.vehicle_params = None
        self.sim_params = None
        self.degradation_calc = None
    def reset(self, vehicle_config: Dict, sim_config: Dict) -> np.ndarray:
        self.current_hour = 0
        self.vehicle_params = vehicle_config
        self.sim_params = sim_config
        self.degradation_model_type = self.vehicle_params['degradation_model']
        self.degradation_calc = BatteryDegradationModel(self.vehicle_params)
        self.current_soc = self.sim_params['initial_soc']
        self.current_prices = random.choice(self.daily_profiles)
        return self._get_state()
    def _get_state(self) -> np.ndarray:
        price = self.current_prices.get(self.current_hour, 0)
        norm_hour = self.current_hour / 23.0
        norm_price = min(price / self.max_price_for_norm, 1.0)
        return np.array([norm_hour, self.current_soc, norm_price], dtype=np.float32)
    def _calculate_degradation_cost(self, soc_start: float, soc_end: float, energy_kwh: float) -> float:
        if self.degradation_model_type == 'lfp': return self.degradation_calc.cost_lfp_model(energy_kwh)
        elif self.degradation_model_type == 'nca': return self.degradation_calc.cost_nca_model(soc_start, soc_end)
        else: return self.degradation_calc.cost_simple_linear(energy_kwh)
    def _calculate_reward(self, revenue, cost, degradation, soc_end, action, soc_start) -> float:
        reward = revenue - cost - degradation
        if soc_end < self.sim_params['soc_min_utente']: reward -= self.sim_params['penalita_ansia'] * (self.sim_params['soc_min_utente'] - soc_end) * 100
        action_str = self.actions_map[action]
        if (action_str == 'Carica' and soc_start >= self.vehicle_params['soc_max']) or (action_str == 'Scarica' and soc_start <= self.vehicle_params['soc_min_batteria']): reward -= 0.1
        done = self.current_hour >= 23
        if done:
            target_soc = self.sim_params['soc_target_finale']
            if soc_end < target_soc: reward -= self.sim_params['penalita_ansia'] * 20 * (target_soc - soc_end) * 100
            else: reward += 5.0
        return reward

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        soc_start = self.current_soc; price = self.current_prices.get(self.current_hour, 0); power_kwh = 0
        if self.actions_map[action] == 'Carica' and soc_start < self.vehicle_params['soc_max']: power_kwh = self.vehicle_params['p_carica']
        elif self.actions_map[action] == 'Scarica' and soc_start > self.vehicle_params['soc_min_batteria']: power_kwh = -self.vehicle_params['p_scarica']
        energy_cost, energy_revenue, energy_processed_kwh = 0, 0, 0; soc_end = soc_start
        if power_kwh > 0:
            energy_stored = power_kwh * self.vehicle_params['efficienza_carica']; soc_end += energy_stored / self.vehicle_params['capacita']
            energy_cost = price * power_kwh; energy_processed_kwh = energy_stored
        elif power_kwh < 0:
            energy_drawn = -power_kwh / self.vehicle_params['efficienza_scarica']; soc_end -= energy_drawn / self.vehicle_params['capacita']
            energy_revenue = price * -power_kwh; energy_processed_kwh = -energy_drawn
        soc_end = np.clip(soc_end, 0, 1); degradation_cost = self._calculate_degradation_cost(soc_start, soc_end, energy_processed_kwh)
        self.current_hour += 1
        reward = self._calculate_reward(energy_revenue, energy_cost, degradation_cost, soc_end, action, soc_start)
        self.current_soc = soc_end; done = self.current_hour >= 24
        return self._get_state(), reward, done

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

class DQNAgent:
    def __init__(self, device):
        self.device = device
        self.state_size = DQN_PARAMS['state_size']
        self.action_size = DQN_PARAMS['action_size']
        self.buffer_size = DQN_PARAMS['buffer_size']
        self.batch_size = DQN_PARAMS['batch_size']
        self.gamma = DQN_PARAMS['gamma']
        self.lr = DQN_PARAMS['learning_rate']
        self.epsilon = DQN_PARAMS['epsilon_start']
        self.policy_net = QNetwork(self.state_size, self.action_size, DQN_PARAMS['hidden_size']).to(self.device)
        self.target_net = QNetwork(self.state_size, self.action_size, DQN_PARAMS['hidden_size']).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = deque(maxlen=self.buffer_size)
        self.reward_scaler = RewardScaler(DQN_PARAMS['reward_scale']) # Inizializza il RewardScaler
        self.update_target_network()
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_policy_state_dict(self):
        return self.policy_net.state_dict()

    def load_policy_state_dict(self, state_dict):
        self.policy_net.load_state_dict(state_dict)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def choose_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon: return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad(): q_values = self.policy_net(state_tensor)
        return torch.argmax(q_values).item()
    def learn(self):
        if len(self.memory) < self.batch_size: return
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device) # Le reward sono già scalate
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).unsqueeze(1).to(self.device)
        current_q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.gamma * next_q_values * (~dones))
        loss = nn.MSELoss()(current_q_values, target_q_values);
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
    def save_model(self, path: str):
        torch.save(self.policy_net.state_dict(), path)
        print(f"\nModello salvato in '{path}'")

class RewardScaler:
    def __init__(self, scale_factor: float):
        self.scale_factor = scale_factor

    def scale(self, reward: float) -> float:
        return reward * self.scale_factor

    def unscale(self, scaled_reward: float) -> float:
        return scaled_reward / self.scale_factor

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

def create_daily_profiles(df: pd.DataFrame, target_zone: str, num_days: int):
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

def worker_process(worker_id, env_config_queue, experience_queue, policy_update_queue, daily_price_profiles, device, worker_batch_size, episode_result_queue):
    print(f"Worker {worker_id}: Avviato su {device}")
    env = V2G_Environment(daily_price_profiles) # Inizializza con i profili di prezzo
    
    policy_net = QNetwork(DQN_PARAMS['state_size'], DQN_PARAMS['action_size'], DQN_PARAMS['hidden_size']).to(device)
    policy_net.eval()

    # Inizializza con una policy vuota, verrà aggiornata dal learner
    current_policy_state_dict = None
    experience_buffer = [] # Buffer per accumulare esperienze
    reward_scaler = RewardScaler(DQN_PARAMS['reward_scale']) # Inizializza il RewardScaler

    while True:
        # Controlla se c'è un segnale di STOP dal learner
        if not env_config_queue.empty():
            msg = env_config_queue.get()
            if msg == "STOP":
                print(f"Worker {worker_id}: Ricevuto segnale di STOP. Terminazione.")
                break
        
        # Aggiorna la policy del worker con l'ultima versione dal learner
        if not policy_update_queue.empty():
            current_policy_state_dict = policy_update_queue.get()
            policy_net.load_state_dict(current_policy_state_dict)
            # print(f"Worker {worker_id}: Policy aggiornata.")

        # Genera una nuova configurazione per ogni episodio
        # I worker ricevono la configurazione specifica per il loro ambiente
        if not env_config_queue.empty():
            vehicle_config, sim_config = env_config_queue.get()
        else:
            # Se la coda è vuota, il worker attende una nuova configurazione
            # Questo può accadere se il learner non sta producendo abbastanza configurazioni
            # o se i worker sono troppo veloci. Per ora, usiamo una configurazione di default
            # o attendiamo. Per un training continuo, la coda non dovrebbe essere vuota a lungo.
            continue # Attendi la prossima configurazione

        state = env.reset(vehicle_config, sim_config)
        done = False
        episode_total_reward = 0 # Inizializza il reward totale per l'episodio

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = policy_net(state_tensor)
            action = torch.argmax(q_values).item()

            next_state, reward, done = env.step(action)
            episode_total_reward += reward # Accumula il reward non scalato
            scaled_reward = reward_scaler.scale(reward) # Scala la reward
            experience_buffer.append((state, action, scaled_reward, next_state, done))
            state = next_state

            # Invia un batch di esperienze quando il buffer è pieno
            if len(experience_buffer) >= worker_batch_size:
                experience_queue.put(experience_buffer)
                experience_buffer = [] # Resetta il buffer

        # Invia le esperienze rimanenti alla fine dell'episodio
        if experience_buffer:
            experience_queue.put(experience_buffer)
            experience_buffer = []
        
        # Invia il reward totale dell'episodio al learner
        episode_result_queue.put(episode_total_reward)

    print(f"Worker {worker_id}: Processo terminato.")


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

def create_daily_profiles(df: pd.DataFrame, target_zone: str, num_days: int):
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
# CICLO DI ADDESTRAMENTO
# ========================================================================

def run_training(agent: DQNAgent, training_profiles: List[Dict], model_path: str, device, profile_name: str, chem_name: str):
    print(f"\n--- AVVIO ADDESTRAMENTO PER MODELLO '{model_path}' ---")
    total_rewards = []
    num_episodes = DQN_PARAMS['episodes']
    best_avg_reward = -np.inf
    patience = DQN_PARAMS.get('early_stopping_patience', 100)
    min_delta = DQN_PARAMS.get('early_stopping_min_delta', 0.01)
    patience_counter = 0

    # Code per la comunicazione tra learner e worker
    num_workers = DQN_PARAMS['num_workers'] # Usa il parametro configurato
    env_config_queue = multiprocessing.Queue() # Per inviare configurazioni agli ambienti dei worker
    experience_queue = multiprocessing.Queue() # Per ricevere esperienze dai worker
    policy_update_queue = multiprocessing.Queue() # Per inviare aggiornamenti della policy ai worker
    episode_result_queue = multiprocessing.Queue() # Per ricevere i risultati degli episodi dai worker

    workers = []
    try:
        for i in range(num_workers):
            p = multiprocessing.Process(target=worker_process, args=(i, env_config_queue, experience_queue, policy_update_queue, training_profiles, device, DQN_PARAMS['worker_batch_size'], episode_result_queue))
            workers.append(p)
            p.start()

        # Invia la policy iniziale ai worker
        policy_update_queue.put(agent.get_policy_state_dict())

        print(f"Avviati {num_workers} worker. In attesa delle prime esperienze...")

        for episode in range(num_episodes):
            # Invia una nuova configurazione ai worker per ogni episodio
            sim_config = {**USER_PROFILES[profile_name]}
            vehicle_config = {**BASE_VEHICLE_PARAMS, **BATTERY_CHEMISTRIES[chem_name]}
            env_config_queue.put((vehicle_config, sim_config))

            # Raccogli i risultati degli episodi completati dai worker
            while not episode_result_queue.empty():
                total_rewards.append(episode_result_queue.get())

            # Il learner apprende dalle esperienze raccolte dai worker
            # Prendi un batch di esperienze dalla coda se disponibile
            while not experience_queue.empty():
                batch_of_experiences = experience_queue.get()
                for exp in batch_of_experiences:
                    agent.remember(*exp)
            
            if len(agent.memory) >= agent.batch_size:
                agent.learn()
                
                # Aggiorna la target network e invia la policy aggiornata ai worker periodicamente
                if (episode + 1) % DQN_PARAMS['target_update_freq'] == 0:
                    agent.update_target_network()
                    # Invia la policy aggiornata a tutti i worker
                    updated_policy = agent.get_policy_state_dict()
                    for _ in range(num_workers):
                        policy_update_queue.put(updated_policy)
            
            # La logica di early stopping rimane nel learner
            if (episode + 1) % 100 == 0:
                # Unscale the rewards for meaningful monitoring
                # Assicurati di avere abbastanza episodi per calcolare la media
                if len(total_rewards) >= 100:
                    unscaled_rewards = [agent.reward_scaler.unscale(r) for r in total_rewards[-100:]]
                    avg_reward = np.mean(unscaled_rewards)
                    print(f"Episodio {episode + 1}/{num_episodes}, Avg Reward (100 ep): {avg_reward:.2f}, Epsilon: {agent.epsilon:.4f}")
                    
                    improving = avg_reward > best_avg_reward + min_delta

                    if improving:
                        best_avg_reward = avg_reward
                        agent.save_model(model_path)
                        patience_counter = 0
                        print(f"  -> Nuovo record di Avg Reward! Pazienza resettata.")
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        print(f"\nATTENZIONE: Nessun miglioramento significativo per {patience * 100} episodi. Interruzione anticipata.")
                        break
                else:
                    print(f"Episodio {episode + 1}/{num_episodes}: Non abbastanza episodi completati per calcolare la media (richiesti 100).")
        
    finally:
        # Invia segnale di STOP ai worker e attendi la loro terminazione
        for _ in range(num_workers):
            env_config_queue.put("STOP")
        for p in workers:
            p.join()
        
        # Chiudi le code per rilasciare le risorse
        env_config_queue.close()
        env_config_queue.join_thread()
        experience_queue.close()
        experience_queue.join_thread()
        policy_update_queue.close()
        policy_update_queue.join_thread()
        episode_result_queue.close()
        episode_result_queue.join_thread()

    print(f"--- Addestramento completato! Modello migliore salvato in '{model_path}' con Avg Reward: {best_avg_reward:.2f} ---")

# ========================================================================
# ESECUZIONE AUTOMATICA
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
    print("--- TRAINER AUTOMATICO PER AGENTI DQN V2G ---")
    print("Controllo e addestramento di tutti i modelli mancanti...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo di calcolo in uso: {device}")

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
            model_path = f"dqn_model_{profile_name}_{chem_name}.pth"
            print("" + "-"*60)
            print(f"Verifica combinazione: Profilo '{profile_name.capitalize()}', Batteria '{chem_name.upper()}'")
            print(f"File modello target: '{model_path}'")

            if os.path.exists(model_path):
                print(f"STATO: Trovato modello esistente '{model_path}'.")
                print("1) Salta addestramento")
                print("2) Continua addestramento (Fine-Tune)")
                print("3) Ricomincia da zero (Sovrascrivi)")
                choice = get_user_input("Scegli un'opzione per questo modello [1-3]: ", str, lambda x: x in ['1', '2', '3'])
                
                if choice == '1':
                    print("Addestramento saltato.")
                    skipped_count += 1
                    continue
                elif choice == '2':
                    try:
                        agent = DQNAgent(device)
                        agent.load_model(model_path)
                        print("Modello caricato per fine-tuning.")
                    except Exception as e:
                        print(f"ERRORE: Impossibile caricare il modello. Errore: {e}", file=sys.stderr)
                        print("L'addestramento partirà da zero.", file=sys.stderr)
                        agent = DQNAgent(device)
                elif choice == '3':
                    print("Il modello esistente sarà sovrascritto al termine.")
                    agent = DQNAgent(device)
            else:
                print("STATO: Non trovato. Avvio addestramento...")
                agent = DQNAgent(device)

            trained_count += 1
            run_training(agent, training_profiles, model_path, device, profile_name, chem_name)

    print("" + "="*60)
    print("PROCESSO DI ADDESTRAMENTO DI MASSA COMPLETATO")
    print(f"  - Modelli addestrati in questa sessione: {trained_count}")
    print(f"  - Modelli già esistenti e saltati: {skipped_count}")
    print(f"  - Totale modelli verificati: {trained_count + skipped_count}")
    print("="*60)

if __name__ == "__main__":
    # Questo è fondamentale per il multiprocessing su Windows
    multiprocessing.freeze_support()
    main()
