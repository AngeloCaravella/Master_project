# =======================================================================
# V2G DQN GENERALIST TRAINER
# Author: Angelo Caravella
# Version: 2.6 (Specific Curriculum)
# Description: Semplificato il curriculum per insegnare 3 abilità distinte:
#              carica, scarica SEMPRE vincolata (min > target), e
#              mantenimento stretto. Ripristinata la logica parallela.
# ========================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import os
import sys
from collections import deque, defaultdict
from typing import Dict, List, Tuple
import multiprocessing

# ========================================================================
# CONFIGURAZIONI PREDEFINITE
# ========================================================================
USER_PROFILES_SOURCES = {
    'conservativo': {'penalita_ansia': 0.02}, 'bilanciato': {'penalita_ansia': 0.01}, 'aggressivo': {'penalita_ansia': 0.005}
}
BATTERY_CHEMISTRIES = {
    'nca': {'degradation_model': 'nca', 'costo_batteria': 120}, 'lfp': {'degradation_model': 'lfp', 'costo_batteria': 90}, 'semplice': {'degradation_model': 'simple', 'costo_batteria': 70}
}
BASE_VEHICLE_PARAMS = {
    'capacita': 60, 'p_carica': 7.4, 'p_scarica': 5.0, 'efficienza_carica': 0.95, 'efficienza_scarica': 0.95, 'soc_max': 0.9, 'soc_min_batteria': 0.1, 'lfp_k_slope': 0.0035,
}
DQN_PARAMS = {
    'state_size': 5, 'action_size': 3, 'hidden_size': 128, 'buffer_size': 100000, 'batch_size': 256,
    'gamma': 0.99, 'learning_rate': 0.001,
    'epsilon_start': 0.03, 'target_update_freq': 10,
    'reward_scale': 0.01,
    'PRICE_ZONE_COLUMN': 'Nord', 'NUM_TRAINING_DAYS': 9, # Ripristinato a 9 (3 scenari x 3 chimiche)

    'USE_DETERMINISTIC_CURRICULUM': True,
    'CURRICULUM_EPISODES_PER_SCENARIO': 2500, # Aumentato di nuovo per addestrare bene i 9 scenari
    'CURRICULUM_MASTERY_THRESHOLD': 6.0,
    'MAX_RETRIES_PER_SCENARIO': 3,
    
    'num_workers': 4,
    'worker_batch_size': 64,
}
MODEL_PATH = "dqn_model_generalist.pth"

# <<< MODIFICA: SCENARIO_SEQUENCE con 3 profili chiari >>>
SCENARIO_SEQUENCE = [
    # LFP
    {'chem': 'lfp', 'profile': 'carica', 'desc': 'LFP - Carica (SoC Basso -> Alto)'},
    {'chem': 'lfp', 'profile': 'mantenimento', 'desc': 'LFP - Mantenimento Stretto'},
    {'chem': 'lfp', 'profile': 'scarica', 'desc': 'LFP - Scarica Vincolata (Min Utente > Target)'},
    # NCA
    {'chem': 'nca', 'profile': 'carica', 'desc': 'NCA - Carica (SoC Basso -> Alto)'},
    {'chem': 'nca', 'profile': 'mantenimento', 'desc': 'NCA - Mantenimento Stretto'},
    {'chem': 'nca', 'profile': 'scarica', 'desc': 'NCA - Scarica Vincolata (Min Utente > Target)'},
    # Semplice
    {'chem': 'semplice', 'profile': 'carica', 'desc': 'Semplice - Carica (SoC Basso -> Alto)'},
    {'chem': 'semplice', 'profile': 'mantenimento', 'desc': 'Semplice - Mantenimento Stretto'},
    {'chem': 'semplice', 'profile': 'scarica', 'desc': 'Semplice - Scarica Vincolata (Min Utente > Target)'},
]

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

class V2G_Environment:
    def __init__(self, daily_price_profiles: List[Dict[int, float]]):
        self.daily_profiles = daily_price_profiles; self.actions_map = {0: 'Attesa', 1: 'Carica', 2: 'Scarica'}; self.max_price_for_norm = 0.5
        self.vehicle_params, self.sim_params, self.degradation_calc = None, None, None
    def reset(self, dynamic_vehicle_config: Dict, dynamic_sim_params: Dict) -> np.ndarray:
        self.current_hour = 0; self.vehicle_params = dynamic_vehicle_config; self.sim_params = dynamic_sim_params
        self.degradation_model_type = self.vehicle_params['degradation_model']; self.degradation_calc = BatteryDegradationModel(self.vehicle_params)
        self.current_soc = self.sim_params['initial_soc']; self.current_prices = random.choice(self.daily_profiles)
        return self._get_state()
    def _get_state(self) -> np.ndarray:
        price = self.current_prices.get(self.current_hour, 0); norm_hour = self.current_hour / 23.0
        norm_price = min(price / self.max_price_for_norm, 1.0)
        return np.array([norm_hour, self.current_soc, norm_price, self.sim_params['soc_min_utente'], self.sim_params['soc_target_finale']], dtype=np.float32)
    def _calculate_degradation_cost(self, soc_start: float, soc_end: float, energy_kwh: float) -> float:
        if self.degradation_model_type == 'lfp': return self.degradation_calc.cost_lfp_model(energy_kwh)
        elif self.degradation_model_type == 'nca': return self.degradation_calc.cost_nca_model(soc_start, soc_end)
        else: return self.degradation_calc.cost_simple_linear(energy_kwh)
    def _calculate_reward(self, revenue, cost, degradation, soc_end, action, soc_start) -> float:
        reward = revenue - cost - degradation
        if soc_end < self.sim_params['soc_min_utente']: reward -= self.sim_params['penalita_ansia'] * (self.sim_params['soc_min_utente'] - soc_end) * 100
        action_str = self.actions_map[action]
        if (action_str == 'Carica' and soc_start >= self.vehicle_params['soc_max']) or (action_str == 'Scarica' and soc_start <= self.vehicle_params['soc_min_batteria']): reward -= 5.0
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
        super(QNetwork, self).__init__(); self.fc1 = nn.Linear(state_size, hidden_size); self.fc2 = nn.Linear(hidden_size, hidden_size); self.fc3 = nn.Linear(hidden_size, action_size)
    def forward(self, x: torch.Tensor) -> torch.Tensor: x = torch.relu(self.fc1(x)); x = torch.relu(self.fc2(x)); return self.fc3(x)

class DQNAgent:
    def __init__(self, device):
        self.device = device; self.state_size = DQN_PARAMS['state_size']; self.action_size = DQN_PARAMS['action_size']; self.buffer_size = DQN_PARAMS['buffer_size']; self.batch_size = DQN_PARAMS['batch_size']; self.gamma = DQN_PARAMS['gamma']; self.lr = DQN_PARAMS['learning_rate']; self.epsilon = DQN_PARAMS['epsilon_start']
        self.policy_net = QNetwork(self.state_size, self.action_size, DQN_PARAMS['hidden_size']).to(self.device); self.target_net = QNetwork(self.state_size, self.action_size, DQN_PARAMS['hidden_size']).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr); self.memory = deque(maxlen=self.buffer_size); self.reward_scaler = RewardScaler(DQN_PARAMS['reward_scale']); self.update_target_network()
    def update_target_network(self): self.target_net.load_state_dict(self.policy_net.state_dict())
    def get_policy_state_dict(self): return self.policy_net.state_dict()
    def load_policy_state_dict(self, state_dict): self.policy_net.load_state_dict(state_dict)
    def remember(self, state, action, reward, next_state, done): self.memory.append((state, action, reward, next_state, done))
    def choose_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon: return random.randrange(self.action_size)
        with torch.no_grad(): return torch.argmax(self.policy_net(torch.FloatTensor(state).unsqueeze(0).to(self.device))).item()
    def learn(self):
        if len(self.memory) < self.batch_size: return
        minibatch = random.sample(self.memory, self.batch_size); states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.FloatTensor(np.array(states)).to(self.device); actions = torch.LongTensor(actions).unsqueeze(1).to(self.device); rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device); next_states = torch.FloatTensor(np.array(next_states)).to(self.device); dones = torch.BoolTensor(dones).unsqueeze(1).to(self.device)
        current_q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad(): next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1); target_q_values = rewards + (self.gamma * next_q_values * (~dones))
        loss = nn.MSELoss()(current_q_values, target_q_values); self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
    def save_model(self, path: str): torch.save(self.policy_net.state_dict(), path); print(f"\nModello salvato in '{path}'")
    def load_model(self, path: str):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.update_target_network()
        print(f"Modello caricato con successo da '{path}'")

class RewardScaler:
    def __init__(self, scale_factor: float): self.scale_factor = scale_factor
    def scale(self, reward: float) -> float: return reward * self.scale_factor
    def unscale(self, scaled_reward: float) -> float: return scaled_reward / self.scale_factor

def load_price_data(file_path: str = "downloads/PrezziZonali.xlsx"):
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
    num_profiles_needed = num_days * (DQN_PARAMS['MAX_RETRIES_PER_SCENARIO'] + 1)
    selected_profiles = all_profiles_from_zone[:min(num_profiles_needed, len(all_profiles_from_zone))]
    print(f"INFO: Trovati {len(all_profiles_from_zone)} giorni. Selezionati {len(selected_profiles)} profili unici per training e tentativi.")
    if len(selected_profiles) < num_days:
        print("ATTENZIONE: Non ci sono abbastanza profili di prezzo unici per tutti i giorni di training. Alcuni profili verranno riutilizzati.", file=sys.stderr)
    return selected_profiles

# <<< MODIFICA: Funzione aggiornata con la nuova logica di curriculum >>>
def generate_deterministic_episode_config(scenario: Dict):
    chem_name = scenario['chem']; soc_profile = scenario['profile']
    vehicle_config = {**BASE_VEHICLE_PARAMS, **BATTERY_CHEMISTRIES[chem_name]}
    random_profile_name = random.choice(list(USER_PROFILES_SOURCES.keys()))
    soc_min_batt, soc_max_batt = BASE_VEHICLE_PARAMS['soc_min_batteria'], BASE_VEHICLE_PARAMS['soc_max']
    
    soc_basso_range = (soc_min_batt + 0.05, soc_min_batt + 0.25)
    soc_medio_range = (soc_min_batt + 0.3, soc_max_batt - 0.3)
    soc_alto_range = (soc_max_batt - 0.25, soc_max_batt - 0.05)

    if soc_profile == 'carica':
        initial_soc = round(random.uniform(*soc_basso_range), 2)
        soc_target_finale = round(random.uniform(*soc_alto_range), 2)
        soc_min_utente = round(random.uniform(soc_min_batt, initial_soc), 2)

    elif soc_profile == 'scarica': # Scarica SEMPRE vincolata
        initial_soc = round(random.uniform(*soc_alto_range), 2)
        soc_target_finale = round(random.uniform(*soc_basso_range), 2)
        soc_min_utente = round(random.uniform(soc_target_finale + 0.1, initial_soc), 2) # Garantisce min > target

    elif soc_profile == 'mantenimento': # Mantenimento "stretto"
        initial_soc = round(random.uniform(*soc_medio_range), 2)
        soc_target_finale = round(random.uniform(initial_soc - 0.05, initial_soc + 0.05), 2)
        soc_min_utente = round(random.uniform(min(initial_soc, soc_target_finale) - 0.1, min(initial_soc, soc_target_finale)), 2)

    else: # Fallback
        initial_soc = round(random.uniform(soc_min_batt, soc_max_batt), 2)
        soc_min_utente = round(random.uniform(soc_min_batt, initial_soc), 2)
        soc_target_finale = round(random.uniform(soc_min_utente, soc_max_batt), 2)

    # Clipping di sicurezza finale per garantire la coerenza
    initial_soc = np.clip(initial_soc, soc_min_batt, soc_max_batt)
    soc_min_utente = np.clip(soc_min_utente, soc_min_batt, soc_max_batt - 0.01)
    soc_target_finale = np.clip(soc_target_finale, soc_min_batt, soc_max_batt)
    
    sim_params = {'initial_soc': initial_soc, 'soc_min_utente': soc_min_utente, 'soc_target_finale': soc_target_finale, 'penalita_ansia': USER_PROFILES_SOURCES[random_profile_name]['penalita_ansia']}
    return vehicle_config, sim_params

def worker_process(worker_id, scenario_queue, experience_queue, policy_update_queue, device):
    print(f"Worker {worker_id}: Avviato su {device}")
    env = V2G_Environment([])
    policy_net = QNetwork(DQN_PARAMS['state_size'], DQN_PARAMS['action_size'], DQN_PARAMS['hidden_size']).to(device)
    policy_net.eval()
    reward_scaler = RewardScaler(DQN_PARAMS['reward_scale'])
    epsilon = DQN_PARAMS['epsilon_start']

    while True:
        if not policy_update_queue.empty():
            policy_state_dict = policy_update_queue.get()
            policy_net.load_state_dict(policy_state_dict)
        
        msg = scenario_queue.get()
        if msg == "STOP":
            print(f"Worker {worker_id}: Ricevuto segnale STOP."); break
        
        current_scenario, current_day_profile = msg
        env.daily_profiles = current_day_profile
        
        experience_buffer = []
        vehicle_config, sim_params = generate_deterministic_episode_config(current_scenario)
        state, done, episode_total_reward = env.reset(vehicle_config, sim_params), False, 0
        
        while not done:
            if random.random() < epsilon:
                action = random.randrange(DQN_PARAMS['action_size'])
            else:
                with torch.no_grad():
                    action = policy_net(torch.FloatTensor(state).unsqueeze(0).to(device)).argmax().item()
            
            next_state, reward, done = env.step(action)
            episode_total_reward += reward
            scaled_reward = reward_scaler.scale(reward)
            experience_buffer.append((state, action, scaled_reward, next_state, done))
            state = next_state
        
        experience_queue.put((experience_buffer, episode_total_reward))

def run_training(agent: DQNAgent, model_path: str, device, training_profiles: List[Dict], num_training_days: int, is_fine_tuning: bool):
    # ... (La logica del learner rimane identica alla versione 2.5) ...
    print("\n--- AVVIO ADDESTRAMENTO A CURRICULUM DETERMINISTICO (PARALLEL) ---")
    
    num_workers = DQN_PARAMS['num_workers']
    scenario_queue = multiprocessing.Queue()
    experience_queue = multiprocessing.Queue()
    policy_update_queue = multiprocessing.Queue(maxsize=num_workers)

    workers = []
    for i in range(num_workers):
        # Passiamo una lista vuota di profili, tanto verranno aggiornati dinamicamente
        p = multiprocessing.Process(target=worker_process, args=(i, scenario_queue, experience_queue, policy_update_queue, device))
        workers.append(p)
        p.start()

    try:
        mastery_threshold = DQN_PARAMS['CURRICULUM_MASTERY_THRESHOLD']
        episodes_per_scenario = DQN_PARAMS['CURRICULUM_EPISODES_PER_SCENARIO']
        total_episodes_ran = 0
        day_index = 0
        profile_idx = 0
        retry_counts = defaultdict(int)
        max_retries = DQN_PARAMS['MAX_RETRIES_PER_SCENARIO']

        while day_index < num_training_days:
            current_scenario = SCENARIO_SEQUENCE[day_index % len(SCENARIO_SEQUENCE)]
            current_day_profile = [training_profiles[profile_idx % len(training_profiles)]]
            
            print("\n" + "="*80)
            current_attempt = retry_counts[day_index] + 1
            print(f"GIORNO {day_index + 1}/{num_training_days} - SCENARIO: {current_scenario['desc']} (Tentativo {current_attempt}/{max_retries+1})")
            print("="*80)

            for _ in range(episodes_per_scenario):
                scenario_queue.put((current_scenario, current_day_profile))
            
            scenario_rewards = []
            
            for episode_num in range(1, episodes_per_scenario + 1):
                exp_batch, total_reward = experience_queue.get()
                for exp in exp_batch: agent.remember(*exp)
                scenario_rewards.append(total_reward)
                total_episodes_ran += 1

                if len(agent.memory) >= agent.batch_size:
                    agent.learn()
                
                if total_episodes_ran % DQN_PARAMS['target_update_freq'] == 0:
                    agent.update_target_network()
                    while not policy_update_queue.empty(): policy_update_queue.get()
                    latest_policy = agent.get_policy_state_dict()
                    for _ in range(num_workers): policy_update_queue.put(latest_policy)

                if episode_num % 100 == 0:
                    avg_reward = np.mean(scenario_rewards[-100:])
                    print(f"\r  Episodio {episode_num}/{episodes_per_scenario}, Avg Reward: {avg_reward:.2f}, Buffer: {len(agent.memory)}/{agent.buffer_size}", end="")
            
            print()
            final_avg_reward = np.mean(scenario_rewards) if scenario_rewards else -1.0
            print(f"Risultato finale per lo scenario: Avg Reward = {final_avg_reward:.2f}")

            if final_avg_reward > 0:
                print("Performance sufficiente. Passo allo scenario successivo.")
                day_index += 1; profile_idx += 1
                agent.save_model(model_path)
            else:
                retry_counts[day_index] += 1
                if retry_counts[day_index] > max_retries:
                    print(f"Limite massimo di tentativi raggiunto. Passo allo scenario successivo.")
                    day_index += 1; profile_idx += 1
                else:
                    print("Performance insufficiente. Riprovo lo stesso scenario.")
                    profile_idx += 1
    
    finally:
        for _ in range(num_workers): scenario_queue.put("STOP")
        for p in workers: p.join()
        print("\nTutti i worker sono stati terminati.")

    print("\n--- CURRICULUM COMPLETATO ---")
    agent.save_model(model_path)


def get_user_input(prompt: str, type_converter, validator, error_message="Input non valido. Riprova."):
    while True:
        try:
            value = type_converter(input(prompt))
            if validator(value): return value
            else: print(error_message)
        except (ValueError, TypeError):
            print("Formato input non valido. Inserire un numero.")

def main():
    # ... (La funzione main rimane identica, ma bisogna chiedere un multiplo di 9) ...
    print("--- TRAINER PER AGENTE DQN GENERALISTA V2G (CURRICULUM DETERMINISTICO V2.6 - PARALLEL) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo di calcolo in uso: {device}")

    agent = DQNAgent(device)
    agent.policy_net.share_memory()

    if os.path.exists(MODEL_PATH):
        print(f"STATO: Trovato modello esistente '{MODEL_PATH}'.")
        print("1) Salta addestramento")
        print("2) Continua addestramento (Fine-Tune)")
        print("3) Ricomincia da zero (Sovrascrivi)")
        choice = get_user_input("Scegli un'opzione [1-3]: ", str, lambda x: x in ['1', '2', '3'])
        
        if choice == '1': print("Addestramento saltato."); return
        elif choice == '2':
            try: agent.load_model(MODEL_PATH); print("Modello caricato per fine-tuning."); start_training, is_fine_tuning = True, True
            except Exception as e: print(f"ERRORE: Impossibile caricare il modello. Errore: {e}", file=sys.stderr); start_training, is_fine_tuning = True, False
        elif choice == '3': print("Il modello esistente sarà sovrascritto al termine."); start_training, is_fine_tuning = True, False
    else:
        print(f"STATO: Nessun modello esistente trovato. L'addestramento partirà da zero."); start_training, is_fine_tuning = True, False

    if start_training:
        price_data = load_price_data()
        available_zones = [col for col in price_data.columns if col not in ['Ora', 'Data']]; print("\nZone di prezzo disponibili:")
        for i, zone in enumerate(available_zones): print(f"  {i+1}) {zone}")
        zone_choice_idx = get_user_input(f"Scegli la zona di prezzo [1-{len(available_zones)}]: ", int, lambda x: 1 <= x <= len(available_zones)) - 1
        target_zone = available_zones[zone_choice_idx]
        
        max_available_days = len(price_data) // 24
        print(f"\nIl curriculum ha una sequenza di 9 scenari.")
        num_days = get_user_input(f"Quanti giorni usare per il training? (multiplo di 9, max {max_available_days}): ", int, lambda x: 1 <= x <= max_available_days and x % 9 == 0, "ERRORE: Inserire un numero di giorni valido e che sia un multiplo di 9.")
        
        training_profiles = create_daily_profiles(price_data, target_zone=target_zone, num_days=num_days)
        if not training_profiles: print("Creazione profili fallita. Uscita.", file=sys.stderr); return
    
        run_training(agent, MODEL_PATH, device, training_profiles, num_days, is_fine_tuning)
    
    print("\nPROCESSO COMPLETATO")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
