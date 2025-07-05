# =======================================================================
# V2G DQN GENERALIST TRAINER
# Author: Angelo Caravella 
# Version: 1.1
# Description: Strumento di addestramento per un singolo agente DQN "generalista",
#              capace di adattarsi a parametri di simulazione dinamici.
#              Lo stato dell'agente è esteso per includere gli obiettivi.
#              Corretta la generazione di configurazioni per evitare contraddizioni logiche.
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

# Non più usati per definire i modelli, ma come "fonti" per la randomizzazione
USER_PROFILES_SOURCES = {
    'conservativo': {'penalita_ansia': 0.02},
    'bilanciato': {'penalita_ansia': 0.01},
    'aggressivo': {'penalita_ansia': 0.005}
}

BATTERY_CHEMISTRIES = {
    'nca': {
        'degradation_model': 'nca',
        'costo_batteria': 120
    },
    'lfp': {
        'degradation_model': 'lfp',
        'costo_batteria': 90
    },
    'semplice': {
        'degradation_model': 'simple',
        'costo_batteria': 70
    }
}

BASE_VEHICLE_PARAMS = {
    'capacita': 60, 'p_carica': 7.4, 'p_scarica': 5.0, 'efficienza_carica': 0.95,
    'efficienza_scarica': 0.95, 'soc_max': 0.9, 'soc_min_batteria': 0.1, 'lfp_k_slope': 0.0035,
}

# Stato esteso a 5 dimensioni!
DQN_PARAMS = {
    'state_size': 5, 'action_size': 3, 'hidden_size': 128, # Rete più grande per un compito più complesso
    'buffer_size': 100000, 'batch_size': 64, 'gamma': 0.99, 'learning_rate': 0.001, 
    'epsilon_start': 0.03, 'target_update_freq': 10,
    'episodes': 30000, # Aumentiamo gli episodi per un addestramento più robusto
    'early_stopping_patience': 100,
    'early_stopping_min_delta': 0.01,
    'worker_batch_size': 64, # Nuovo parametro per il batching delle esperienze dai worker
    'reward_scale': 0.01, # Scala per la normalizzazione delle reward
    'num_workers': 8 # Numero di processi worker
}

MODEL_PATH = "dqn_model_generalist.pth"

# ========================================================================
# CLASSI (Ambiente e Stato modificati)
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

    def reset(self, dynamic_vehicle_config: Dict, dynamic_sim_params: Dict) -> np.ndarray:
        self.current_hour = 0
        self.vehicle_params = dynamic_vehicle_config
        self.sim_params = dynamic_sim_params
        self.degradation_model_type = self.vehicle_params['degradation_model']
        self.degradation_calc = BatteryDegradationModel(self.vehicle_params)
        
        self.current_soc = self.sim_params['initial_soc']
        self.current_prices = random.choice(self.daily_profiles)
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        price = self.current_prices.get(self.current_hour, 0)
        norm_hour = self.current_hour / 23.0
        norm_price = min(price / self.max_price_for_norm, 1.0)
        # Stato esteso con gli obiettivi!
        return np.array([
            norm_hour, 
            self.current_soc, 
            norm_price,
            self.sim_params['soc_min_utente'],
            self.sim_params['soc_target_finale']
        ], dtype=np.float32)

    def _calculate_degradation_cost(self, soc_start: float, soc_end: float, energy_kwh: float) -> float:
        if self.degradation_model_type == 'lfp': return self.degradation_calc.cost_lfp_model(energy_kwh)
        elif self.degradation_model_type == 'nca': return self.degradation_calc.cost_nca_model(soc_start, soc_end)
        else: return self.degradation_calc.cost_simple_linear(energy_kwh)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        soc_start = self.current_soc
        price = self.current_prices.get(self.current_hour, 0)
        power_kwh = 0
        if self.actions_map[action] == 'Carica' and soc_start < self.vehicle_params['soc_max']: power_kwh = self.vehicle_params['p_carica']
        elif self.actions_map[action] == 'Scarica' and soc_start > self.vehicle_params['soc_min_batteria']: power_kwh = -self.vehicle_params['p_scarica']
        
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
        
        anxiety_cost = 0
        if soc_end < self.sim_params['soc_min_utente']: 
            anxiety_cost = self.sim_params['penalita_ansia'] * (self.sim_params['soc_min_utente'] - soc_end) * 100
        
        reward = energy_revenue - energy_cost - degradation_cost - anxiety_cost
        self.current_soc = soc_end
        self.current_hour += 1
        done = self.current_hour >= 23

        if done:
            target_soc = self.sim_params['soc_target_finale']
            if self.current_soc < target_soc: 
                reward -= self.sim_params['penalita_ansia'] * 20 * (target_soc - self.current_soc) * 100
        
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

def create_daily_profiles(df: pd.DataFrame) -> List[Dict]:
    all_zones = [col for col in df.columns if col not in ['Ora', 'Data']]
    all_profiles = []
    for zone in all_zones:
        zone_prices = df[zone].dropna()
        for i in range(0, len(zone_prices), 24):
            if len(zone_prices.iloc[i:i+24]) == 24: all_profiles.append({h: p for h, p in enumerate(zone_prices.iloc[i:i+24])})
    return all_profiles

def generate_random_episode_config() -> Tuple[Dict, Dict]:
    """Genera una configurazione logica e casuale per un singolo episodio."""
    soc_min_batt = BASE_VEHICLE_PARAMS['soc_min_batteria']
    soc_max_batt = BASE_VEHICLE_PARAMS['soc_max']

    # 1. Genera il SoC minimo utente in un range valido.
    soc_min_utente = round(random.uniform(soc_min_batt + 0.05, soc_max_batt - 0.1), 2)
    
    # 2. Genera il SoC target finale in modo che sia SEMPRE >= del SoC minimo utente.
    soc_target_finale = round(random.uniform(soc_min_utente, soc_max_batt), 2)

    # L'initial_soc può essere ovunque nel range della batteria
    initial_soc = round(random.uniform(soc_min_batt, soc_max_batt), 2)

    # Scegli una penalità e una chimica a caso
    random_profile_name = random.choice(list(USER_PROFILES_SOURCES.keys()))
    random_chem_name = random.choice(list(BATTERY_CHEMISTRIES.keys()))

    sim_params = {
        'initial_soc': initial_soc,
        'soc_min_utente': soc_min_utente,
        'soc_target_finale': soc_target_finale,
        'penalita_ansia': USER_PROFILES_SOURCES[random_profile_name]['penalita_ansia']
    }
    
    vehicle_config = {**BASE_VEHICLE_PARAMS, **BATTERY_CHEMISTRIES[random_chem_name]}

    return vehicle_config, sim_params

# ========================================================================
# CICLO DI ADDESTRAMENTO
# ========================================================================

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
        vehicle_config, sim_params = generate_random_episode_config()
        state = env.reset(vehicle_config, sim_params)
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


def run_training(agent: DQNAgent, env: V2G_Environment, model_path: str, device, training_profiles):
    print(f"--- AVVIO ADDESTRAMENTO PER MODELLO GENERALISTA '{model_path}' ---")
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
                
                # La logica di epsilon decay e early stopping rimane nel learner
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
# ESECUZIONE
# ========================================================================

def main():
    print("--- TRAINER PER AGENTE DQN GENERALISTA V2G ---")

    # Determina il device da usare (GPU se disponibile, altrimenti CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo di calcolo in uso: {device}")

    if os.path.exists(MODEL_PATH):
        print(f"STATO: Modello '{MODEL_PATH}' già esistente. Addestramento saltato.")
        print("Per ri-addestrare, cancellare o rinominare il file esistente.")
        return

    price_data = load_price_data()
    training_profiles = create_daily_profiles(price_data)
    
    if not training_profiles: 
        print("Nessun profilo di prezzo valido per il training. Uscita.", file=sys.stderr)
        return

    # L'ambiente principale è solo per i profili di prezzo, non per l'interazione diretta
    env = V2G_Environment(training_profiles)
    agent = DQNAgent(device) # Passa il device all'agente
    
    # Passa il device anche alla funzione di training
    run_training(agent, env, MODEL_PATH, device, training_profiles)

    print("\nPROCESSO COMPLETATO")

if __name__ == "__main__":
    # Questo è fondamentale per il multiprocessing su Windows
    multiprocessing.freeze_support()
    main()
