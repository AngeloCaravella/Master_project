

# =======================================================================
# V2G DQN BATCH TRAINER
# Author: Angelo Caravella & Gemini
# Version: 1.4
# Description: Strumento di addestramento automatico e di massa per agenti DQN.
#              Lo script controlla tutti i modelli mancanti per ogni combinazione
#              di profilo/batteria e li addestra in autonomia.
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

# =======================================================================
# CONFIGURAZIONI PREDEFINITE
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

DQN_PARAMS = {
    'state_size': 3, 'action_size': 3, 'hidden_size': 64, 'buffer_size': 100000,
    'batch_size': 64, 'gamma': 0.98, 'learning_rate': 0.001, 'epsilon_start': 1.0,
    'epsilon_decay': 0.9995, 'epsilon_min': 0.01, 'target_update_freq': 10,
    'episodes': 15000, # Manteniamo un numero di episodi elevato per un buon training
}

# =======================================================================
# CLASSI (Invariate)
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
    def __init__(self, daily_price_profiles: List[Dict[int, float]], vehicle_config: Dict, sim_config: Dict):
        self.vehicle_params = vehicle_config
        self.sim_params = sim_config
        self.degradation_model_type = self.vehicle_params['degradation_model']
        self.degradation_calc = BatteryDegradationModel(self.vehicle_params)
        self.daily_profiles = daily_price_profiles
        self.actions_map = {0: 'Attesa', 1: 'Carica', 2: 'Scarica'}
        self.max_price_for_norm = 0.5

    def reset(self) -> np.ndarray:
        self.current_hour = 0
        self.current_soc = self.sim_params['initial_soc']
        self.current_prices = random.choice(self.daily_profiles)
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        price = self.current_prices.get(self.current_hour, 0)
        norm_hour = self.current_hour / 23.0
        norm_price = min(price / self.max_price_for_norm, 1.0)
        return np.array([norm_hour, self.current_soc, norm_price], dtype=np.float32)

    def _calculate_degradation_cost(self, soc_start: float, soc_end: float, energy_kwh: float) -> float:
        if self.degradation_model_type == 'lfp':
            return self.degradation_calc.cost_lfp_model(energy_kwh)
        elif self.degradation_model_type == 'nca':
            return self.degradation_calc.cost_nca_model(soc_start, soc_end)
        else:
            return self.degradation_calc.cost_simple_linear(energy_kwh)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
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
                reward -= self.sim_params['penalita_ansia'] * 5 * (target_soc - self.current_soc) * 100
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
    def __init__(self):
        self.state_size = DQN_PARAMS['state_size']
        self.action_size = DQN_PARAMS['action_size']
        self.buffer_size = DQN_PARAMS['buffer_size']
        self.batch_size = DQN_PARAMS['batch_size']
        self.gamma = DQN_PARAMS['gamma']
        self.lr = DQN_PARAMS['learning_rate']
        self.epsilon = DQN_PARAMS['epsilon_start']
        self.policy_net = QNetwork(self.state_size, self.action_size, DQN_PARAMS['hidden_size'])
        self.target_net = QNetwork(self.state_size, self.action_size, DQN_PARAMS['hidden_size'])
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = deque(maxlen=self.buffer_size)
        self.update_target_network()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return torch.argmax(q_values).item()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.BoolTensor(dones).unsqueeze(1)
        current_q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.gamma * next_q_values * (~dones))
        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        self.epsilon = max(DQN_PARAMS['epsilon_min'], self.epsilon * DQN_PARAMS['epsilon_decay'])

    def save_model(self, path: str):
        torch.save(self.policy_net.state_dict(), path)
        print(f"\nModello salvato in '{path}'")

# =======================================================================
# FUNZIONI AUSILIARIE (Invariate)
# ========================================================================

def load_price_data(file_path: str = "downloads/PrezziZonali.xlsx") -> pd.DataFrame:
    try:
        df = pd.read_excel(file_path)
        for col in df.columns:
            if col not in ['Ora', 'Data']:
                if df[col].dtype == 'object':
                    df[col] = df[col].str.replace(',', '.', regex=False).astype(float)
                df[col] = df[col] / 1000
        return df
    except FileNotFoundError:
        print(f"ERRORE: File prezzi '{file_path}' non trovato.", file=sys.stderr)
        sys.exit(1)

def create_daily_profiles(df: pd.DataFrame, test_zone: str = "Italia") -> Tuple[List[Dict], List[Dict]]:
    all_zones = [col for col in df.columns if col not in ['Ora', 'Data']]
    training_zones = [z for z in all_zones if z != test_zone]
    training_profiles = []
    for zone in training_zones:
        zone_prices = df[zone].dropna()
        for i in range(0, len(zone_prices), 24):
            if len(zone_prices.iloc[i:i+24]) == 24:
                training_profiles.append({h: p for h, p in enumerate(zone_prices.iloc[i:i+24])})
    return training_profiles, [] # Non ci serve il test set qui

# =======================================================================
# CICLO DI ADDESTRAMENTO
# ========================================================================

def run_training(agent: DQNAgent, env: V2G_Environment, model_path: str):
    print(f"\n--- AVVIO ADDESTRAMENTO PER MODELLO '{model_path}' ---")
    total_rewards = []
    num_episodes = DQN_PARAMS['episodes']
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            episode_reward += reward
        agent.decay_epsilon()
        if (episode + 1) % DQN_PARAMS['target_update_freq'] == 0:
            agent.update_target_network()
        total_rewards.append(episode_reward)
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(total_rewards[-100:])
            print(f"Episodio: {episode + 1}/{num_episodes}, "
                  f"Avg Reward (100 ep): {avg_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.4f}")
    agent.save_model(model_path)
    print(f"--- Addestramento per '{model_path}' completato! ---")

# =======================================================================
# ESECUZIONE AUTOMATICA
# ========================================================================

def main():
    print("--- TRAINER AUTOMATICO PER AGENTI DQN V2G ---")
    print("Controllo e addestramento di tutti i modelli mancanti...")

    price_data = load_price_data()
    training_profiles, _ = create_daily_profiles(price_data)
    
    if not training_profiles:
        print("Nessun profilo di prezzo valido per il training. Uscita.", file=sys.stderr)
        return

    trained_count = 0
    skipped_count = 0

    # Itera su tutte le combinazioni possibili
    for profile_name, profile_params in USER_PROFILES.items():
        for chem_name, chem_params in BATTERY_CHEMISTRIES.items():
            model_path = f"dqn_model_{profile_name}_{chem_name}.pth"
            print("\n" + "-"*60)
            print(f"Verifica combinazione: Profilo '{profile_name.capitalize()}', Batteria '{chem_name.upper()}'")
            print(f"File modello target: '{model_path}'")

            if os.path.exists(model_path):
                print("STATO: Trovato. Addestramento saltato.")
                skipped_count += 1
                continue
            
            print("STATO: Non trovato. Avvio addestramento...")
            trained_count += 1

            # Configura l'ambiente per questa specifica combinazione
            sim_config = {**BASE_SIMULATION_PARAMS, **profile_params}
            vehicle_config = {**BASE_VEHICLE_PARAMS, **chem_params}
            
            env = V2G_Environment(training_profiles, vehicle_config, sim_config)
            agent = DQNAgent()
            
            # Avvia il training
            run_training(agent, env, model_path)

    print("\n" + "="*60)
    print("PROCESSO DI ADDESTRAMENTO DI MASSA COMPLETATO")
    print(f"  - Modelli addestrati in questa sessione: {trained_count}")
    print(f"  - Modelli già esistenti e saltati: {skipped_count}")
    print(f"  - Totale modelli verificati: {trained_count + skipped_count}")
    print("="*60)

if __name__ == "__main__":
    main()
