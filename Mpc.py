import numpy as np
import pandas as pd
import random
import os
import sys
from scipy.optimize import minimize
from typing import Dict, List, Tuple

# ========================================================================
# CONFIGURAZIONE
# ========================================================================

VEHICLE_PARAMS = {
    'capacita': 60,              # kWh
    'p_carica': 7.4,             # kW
    'p_scarica': 5.0,            # kW (potenza di scarica può essere diversa dalla carica)
    'efficienza': 0.92,          #
    'soc_max': 0.9,              # 90%
    'soc_min_batteria': 0.1,     # 10%
    'costo_degradazione': 0.008, # €/kWh - ULTERIORMENTE RIDOTTO, era 0.02
}

SIMULATION_PARAMS = {
    'soc_min_utente': 0.3,       # 30% - Soglia sotto cui si applica la penalità ansia
    'penalita_ansia': 0.005,     # €/% - ULTERIORMENTE RIDOTTO, era 0.01
    'initial_soc': 0.5,          # 50%
    'mpc_horizon': 6,            # ore
    'soc_target_finale': 0.5,    # Target SOC alla fine della giornata per strategie a lungo termine/RL
}

RL_PARAMS = {
    'states_ora': 24,
    'states_soc': 11,            # Da 0 a 10, corrispondente a 0%-100% discretizzato
    'alpha': 0.1,                # Learning rate
    'gamma': 0.98,               # Discount factor - AUMENTATO per maggiore enfasi su ricompense future
    'epsilon': 1.0,              # Initial exploration rate
    'epsilon_decay': 0.9998,     # Decay rate for epsilon - LEGGERMENTE AUMENTATO per convergenza più lenta
    'epsilon_min': 0.01,         # Minimum exploration rate
    'episodes': 100000,          # AUMENTATO per training più robusto, era 50000
    'q_table_file': os.path.join('q_tables', 'q_table_v3.npy') # Cambiato path per versioning
}

# ========================================================================
# CLASSE OTTIMIZZATORE V2G
# ========================================================================

class V2GOptimizer:
    """
    Gestisce le strategie di ottimizzazione per il Vehicle-to-Grid (V2G).
    """
    def __init__(self, vehicle_config: Dict, sim_config: Dict, prices: Dict[int, float]):
        """
        Inizializza l'ottimizzatore V2G.

        Args:
            vehicle_config (Dict): Parametri del veicolo.
            sim_config (Dict): Parametri della simulazione (es. ansia).
            prices (Dict[int, float]): Prezzi dell'energia per ora.
        """
        self.vehicle_params = vehicle_config
        self.sim_params = sim_config
        self.prices = prices
        self.actions_map = {0: 'Attesa', 1: 'Carica', 2: 'Scarica'}

        self.deg_cost_per_kwh = self.vehicle_params['costo_degradazione']
        self.anxiety_penalty_per_perc = self.sim_params['penalita_ansia'] # Rinominato per chiarezza
        self.soc_min_utente = self.sim_params['soc_min_utente']
        self.soc_target_finale = self.sim_params['soc_target_finale']

        self.soc_charge_perc = self._calculate_soc_change(is_charging=True)
        self.soc_discharge_perc = self._calculate_soc_change(is_charging=False)

    def _calculate_soc_change(self, is_charging: bool) -> float:
        """Calcola la variazione percentuale di SOC per carica/scarica."""
        power = self.vehicle_params['p_carica'] if is_charging else self.vehicle_params['p_scarica']
        efficiency = self.vehicle_params['efficienza']
        capacity = self.vehicle_params['capacita']
        
        if is_charging:
            return (power * efficiency / capacity) * 100
        else:
            return (power / efficiency / capacity) * 100

    def _calculate_degradation_cost(self, energy_kwh: float) -> float:
        """Calcola il costo di degradazione in base all'energia processata."""
        return self.deg_cost_per_kwh * abs(energy_kwh)

    def _calculate_anxiety_cost(self, soc: float) -> float:
        """Calcola la penalità per l'ansia se il SOC è sotto il minimo utente."""
        if soc < self.soc_min_utente:
            deficit_soc_perc = (self.soc_min_utente - soc) * 100
            return self.anxiety_penalty_per_perc * deficit_soc_perc
        return 0.0

    def _calculate_terminal_soc_cost(self, soc: float) -> float:
        """
        Calcola il costo/ricompensa per il SOC finale in RL e MPC.
        Penalizza se sotto il target, ricompensa leggermente se sopra, ma è più tollerante.
        Ritorna un valore che VERRÀ SOTTRATTO alla ricompensa RL (quindi positivo = costo, negativo = ricavo).
        """
        target = self.soc_target_finale
        tolerance = 0.08 # Aumentata la tolleranza a +/- 8% del target SOC

        if soc < target - tolerance:
            deficit_perc = (target - soc) * 100
            # Penalità più aggressiva per SOC molto basso, ma coefficiente di ansia ridotto
            return self.anxiety_penalty_per_perc * 10.0 * deficit_perc 
        elif soc > target + tolerance:
            surplus_perc = (soc - target) * 100
            # Ricompensa moderata per SOC sopra il target
            return -self.anxiety_penalty_per_perc * 0.8 * surplus_perc 
        else:
            # Entro il range di tolleranza, diamo un bonus fisso per essere "nel giusto"
            return -0.05 # Piccola ricompensa fissa per essere vicino al target

    def _log_hourly_data(self, log: List, hour: int, soc_start: float, price: float, action: str,
                         energy_cost: float, energy_revenue: float, degradation_cost: float,
                         anxiety_cost: float, net_gain: float, reward_col_name: str):
        """Registra i dati orari in un formato standard."""
        variation_soc = 0
        if action == 'Carica':
            variation_soc = self.soc_charge_perc
        elif action == 'Scarica':
            variation_soc = -self.soc_discharge_perc

        log.append({
            'Ora': hour + 1,
            'SoC Iniziale (%)': round(soc_start * 100, 1),
            'Prezzo (€/kWh)': round(price, 4),
            'Azione': action,
            'Variazione SoC (%)': round(variation_soc, 1), # Arrotondato per visualizzazione
            'Costo Energia (€)': round(energy_cost, 4),
            'Ricavo Energia (€)': round(energy_revenue, 4),
            'Costo Degradazione (€)': round(degradation_cost, 4),
            'Costo Ansia (€)': round(anxiety_cost, 4),
            reward_col_name: round(net_gain, 4)
        })
        
        print(f"{hour+1:3d} | {soc_start*100:11.1f}% | {price:6.4f} | {action:6} | "
              f"{variation_soc:13.1f} | {energy_cost:12.4f} | {energy_revenue:14.4f} | "
              f"{degradation_cost:10.4f} | {anxiety_cost:5.3f} | {net_gain:12.4f}")

    def run_heuristic_strategy(self) -> pd.DataFrame:
        """Esegue la strategia euristica con soglie dinamiche."""
        soc = self.sim_params['initial_soc']
        log = []
        prices = list(self.prices.values())
        avg_price = np.mean(prices)
        std_price = np.std(prices)
        
        # Soglie basate sulla deviazione standard del prezzo
        charge_threshold = avg_price - 0.5 * std_price
        discharge_threshold = avg_price + 0.5 * std_price
        
        cumulative_costs = {'energy': 0, 'revenue': 0, 'degradation': 0, 'anxiety': 0}

        print("\nDettaglio Orario - Strategia Euristica:")
        print("Ora | SoC Iniziale | Prezzo | Azione | Variazione SoC | Costo Energia | Ricavo Energia | Degradazione | Ansia | Guadagno Netto")
        print("----|--------------|--------|--------|---------------|---------------|----------------|--------------|-------|---------------")
        
        for hour, price in self.prices.items():
            soc_start = soc
            action, energy_cost, energy_revenue, degradation_cost = 'Attesa', 0, 0, 0

            # Logica di controllo per non superare i limiti di carica/scarica e SOC
            if price < charge_threshold and soc < self.vehicle_params['soc_max']:
                action = 'Carica'
                energy_to_charge = self.vehicle_params['p_carica'] * self.vehicle_params['efficienza'] # Energia che ENTRA nella batteria
                max_charge_available_kwh = (self.vehicle_params['soc_max'] - soc) * self.vehicle_params['capacita']
                
                actual_energy_stored_kwh = min(energy_to_charge, max_charge_available_kwh)
                actual_energy_purchased_kwh = actual_energy_stored_kwh / self.vehicle_params['efficienza']
                
                soc += actual_energy_stored_kwh / self.vehicle_params['capacita']
                degradation_cost = self._calculate_degradation_cost(actual_energy_stored_kwh)
                energy_cost = price * actual_energy_purchased_kwh
                
            elif price > discharge_threshold and soc > self.vehicle_params['soc_min_batteria']:
                action = 'Scarica'
                energy_to_discharge = self.vehicle_params['p_scarica'] / self.vehicle_params['efficienza'] # Energia che ESCE dalla batteria
                max_discharge_available_kwh = (soc - self.vehicle_params['soc_min_batteria']) * self.vehicle_params['capacita']
                
                actual_energy_drawn_kwh = min(energy_to_discharge, max_discharge_available_kwh)
                actual_energy_delivered_kwh = actual_energy_drawn_kwh * self.vehicle_params['efficienza']
                
                soc -= actual_energy_drawn_kwh / self.vehicle_params['capacita']
                degradation_cost = self._calculate_degradation_cost(actual_energy_drawn_kwh)
                energy_revenue = price * actual_energy_delivered_kwh
            
            anxiety_cost = self._calculate_anxiety_cost(soc)
            net_gain = energy_revenue - energy_cost - degradation_cost - anxiety_cost
            
            cumulative_costs['energy'] += energy_cost
            cumulative_costs['revenue'] += energy_revenue
            cumulative_costs['degradation'] += degradation_cost
            cumulative_costs['anxiety'] += anxiety_cost
            
            soc = np.clip(soc, 0, 1) # Assicurati che SOC rimanga tra 0 e 1
            
            self._log_hourly_data(log, hour, soc_start, price, action, energy_cost, energy_revenue,
                                  degradation_cost, anxiety_cost, net_gain, 'Guadagno Netto Ora (€)')
        
        return self._finalize_log(log, cumulative_costs, 'Guadagno Netto Ora (€)')

    def run_lcvf_strategy(self) -> pd.DataFrame:
        """Esegue la strategia LCVF (Load Conservation Valley-Filling) con V2G."""
        soc = self.sim_params['initial_soc']
        log = []
        cumulative_costs = {'energy': 0, 'revenue': 0, 'degradation': 0, 'anxiety': 0}

        # --- Fase di Pianificazione LCVF con V2G ---
        num_charge_hours = 4  # Le N ore più economiche
        num_discharge_hours = 4 # Le N ore più costose

        sorted_prices = sorted(self.prices.items(), key=lambda item: item[1])
        
        charge_hours = {hour for hour, price in sorted_prices[:num_charge_hours]}
        discharge_hours = {hour for hour, price in sorted_prices[-num_discharge_hours:]}

        plan = {}
        for hour in self.prices.keys():
            if hour in charge_hours:
                plan[hour] = 'Carica'
            elif hour in discharge_hours:
                plan[hour] = 'Scarica'
            else:
                plan[hour] = 'Attesa'
        # --- Fine Pianificazione ---

        print("\nDettaglio Orario - Strategia LCVF (V2G Pianificato):")
        print("Ora | SoC Iniziale | Prezzo | Azione | Variazione SoC | Costo Energia | Ricavo Energia | Degradazione | Ansia | Guadagno Netto")
        print("----|--------------|--------|--------|---------------|---------------|----------------|--------------|-------|---------------")

        for hour, price in self.prices.items():
            soc_start = soc
            action, energy_cost, energy_revenue, degradation_cost = 'Attesa', 0, 0, 0

            planned_action = plan[hour]

            if planned_action == 'Carica' and soc < self.vehicle_params['soc_max']:
                action = 'Carica'
                energy_to_charge = self.vehicle_params['p_carica'] * self.vehicle_params['efficienza']
                max_charge_available_kwh = (self.vehicle_params['soc_max'] - soc) * self.vehicle_params['capacita']
                
                actual_energy_stored_kwh = min(energy_to_charge, max_charge_available_kwh)
                actual_energy_purchased_kwh = actual_energy_stored_kwh / self.vehicle_params['efficienza']
                
                soc += actual_energy_stored_kwh / self.vehicle_params['capacita']
                degradation_cost = self._calculate_degradation_cost(actual_energy_stored_kwh)
                energy_cost = price * actual_energy_purchased_kwh
                
            elif planned_action == 'Scarica' and soc > self.vehicle_params['soc_min_batteria']:
                action = 'Scarica'
                energy_to_discharge = self.vehicle_params['p_scarica'] / self.vehicle_params['efficienza']
                max_discharge_available_kwh = (soc - self.vehicle_params['soc_min_batteria']) * self.vehicle_params['capacita']
                
                actual_energy_drawn_kwh = min(energy_to_discharge, max_discharge_available_kwh)
                actual_energy_delivered_kwh = actual_energy_drawn_kwh * self.vehicle_params['efficienza']
                
                soc -= actual_energy_drawn_kwh / self.vehicle_params['capacita']
                degradation_cost = self._calculate_degradation_cost(actual_energy_drawn_kwh)
                energy_revenue = price * actual_energy_delivered_kwh

            anxiety_cost = self._calculate_anxiety_cost(soc)
            net_gain = energy_revenue - energy_cost - degradation_cost - anxiety_cost
            
            cumulative_costs['energy'] += energy_cost
            cumulative_costs['revenue'] += energy_revenue
            cumulative_costs['degradation'] += degradation_cost
            cumulative_costs['anxiety'] += anxiety_cost
            
            soc = np.clip(soc, 0, 1)
            
            self._log_hourly_data(log, hour, soc_start, price, action, energy_cost, energy_revenue,
                                  degradation_cost, anxiety_cost, net_gain, 'Guadagno Netto Ora (€)')
        
        return self._finalize_log(log, cumulative_costs, 'Guadagno Netto Ora (€)')

    def run_mpc_strategy(self, horizon: int) -> pd.DataFrame:
        """Esegue la strategia MPC con un orizzonte specificato."""
        soc = self.sim_params['initial_soc']
        log = []
        hours = sorted(self.prices.keys())
        
        cumulative_costs = {'energy': 0, 'revenue': 0, 'degradation': 0, 'anxiety': 0}

        print(f"\nDettaglio Orario - Strategia MPC (Orizzonte={horizon}h):")
        print("Ora | SoC Iniziale | Prezzo | Azione | Variazione SoC | Costo Energia | Ricavo Energia | Degradazione | Ansia | Guadagno Netto")
        print("----|--------------|--------|--------|---------------|---------------|----------------|--------------|-------|---------------")
        
        for i, hour in enumerate(hours):
            soc_start = soc
            action, energy_cost, energy_revenue, degradation_cost = 'Attesa', 0, 0, 0
            
            # Determina se applicare il costo terminale nell'ottimizzazione MPC
            # Il costo terminale viene applicato all'ultima ora dell'orizzonte MPC se questa ora è l'ultima della giornata.
            # O se l'orizzonte termina alla fine della giornata.
            is_terminal_for_mpc = (i + horizon >= len(hours))

            horizon_prices = [self.prices[h] for h in hours[i:i+horizon]]
            if len(horizon_prices) > 0:
                action_power = self._solve_mpc(soc, horizon_prices, is_terminal_for_mpc)
                
                # Applica solo la prima azione dell'orizzonte ottimizzato
                if action_power > 0.1: # Carica (soglia per distinguere da attesa)
                    action = 'Carica'
                    power_applied = min(action_power, self.vehicle_params['p_carica'])
                    energy_stored_kwh = power_applied * self.vehicle_params['efficienza']
                    
                    max_charge_kwh = (self.vehicle_params['soc_max'] - soc) * self.vehicle_params['capacita']
                    actual_energy_stored_kwh = min(energy_stored_kwh, max_charge_kwh)
                    actual_energy_purchased_kwh = actual_energy_stored_kwh / self.vehicle_params['efficienza']

                    soc += actual_energy_stored_kwh / self.vehicle_params['capacita']
                    degradation_cost = self._calculate_degradation_cost(actual_energy_stored_kwh)
                    energy_cost = self.prices[hour] * actual_energy_purchased_kwh
                    
                elif action_power < -0.1: # Scarica (soglia per distinguere da attesa)
                    action = 'Scarica'
                    power_applied = min(-action_power, self.vehicle_params['p_scarica']) # power_applied è positivo
                    energy_drawn_kwh = power_applied / self.vehicle_params['efficienza']
                    
                    max_discharge_kwh = (soc - self.vehicle_params['soc_min_batteria']) * self.vehicle_params['capacita']
                    actual_energy_drawn_kwh = min(energy_drawn_kwh, max_discharge_kwh)
                    actual_energy_delivered_kwh = actual_energy_drawn_kwh * self.vehicle_params['efficienza']

                    soc -= actual_energy_drawn_kwh / self.vehicle_params['capacita']
                    degradation_cost = self._calculate_degradation_cost(actual_energy_drawn_kwh)
                    energy_revenue = self.prices[hour] * actual_energy_delivered_kwh
            
            anxiety_cost = self._calculate_anxiety_cost(soc)
            net_gain = energy_revenue - energy_cost - degradation_cost - anxiety_cost
            
            cumulative_costs['energy'] += energy_cost
            cumulative_costs['revenue'] += energy_revenue
            cumulative_costs['degradation'] += degradation_cost
            cumulative_costs['anxiety'] += anxiety_cost
            
            soc = np.clip(soc, 0, 1)
            
            # CORREZIONE DELL'ERRORE: Aggiunto 'soc_start' come terzo argomento
            self._log_hourly_data(log, hour, soc_start, self.prices[hour], action, energy_cost, 
                                  energy_revenue, degradation_cost, anxiety_cost, net_gain, 'Guadagno Netto Ora (€)')

        # Alla fine della giornata, aggiungi la componente terminale al costo totale
        # Questo assicura che il costo terminale sia incluso nel riepilogo finale anche se MPC non lo ha "visto" come terminale nell'ultima ora
        final_soc_cost_summary = self._calculate_terminal_soc_cost(soc)
        cumulative_costs['anxiety'] += final_soc_cost_summary 
        
        return self._finalize_log(log, cumulative_costs, 'Guadagno Netto Ora (€)')

    def _solve_mpc(self, current_soc: float, horizon_prices: List[float], is_terminal: bool) -> float:
        """Risolve il problema di ottimizzazione MPC per l'orizzonte dato."""
        n = len(horizon_prices)
        
        def objective(x):
            total_cost = 0
            soc = current_soc
            
            for i in range(n):
                power = x[i] # kW, positivo per carica, negativo per scarica
                
                # Calcolo del prossimo SOC basato sull'azione proposta
                if power > 0:  # Carica
                    energy_purchased = power # Potenza in kW, usata per calcolare energia acquistata
                    energy_stored_kwh = energy_purchased * self.vehicle_params['efficienza']
                    
                    # Penalità forte se si tenta di caricare oltre soc_max
                    # Aggiunto un termine che limita la carica effettiva per non superare il soc_max
                    max_charge_available_kwh = (self.vehicle_params['soc_max'] - soc) * self.vehicle_params['capacita']
                    actual_energy_stored_kwh = min(energy_stored_kwh, max_charge_available_kwh)
                    
                    # Penalità per tentare di caricare più del consentito
                    if energy_stored_kwh > max_charge_available_kwh + 1e-6: # Aggiungi una piccola tolleranza
                        total_cost += 1e6 * (energy_stored_kwh - max_charge_available_kwh)

                    actual_energy_purchased_kwh = actual_energy_stored_kwh / self.vehicle_params['efficienza']
                    degradation_cost = self._calculate_degradation_cost(actual_energy_stored_kwh)
                    
                    total_cost += horizon_prices[i] * actual_energy_purchased_kwh + degradation_cost
                    soc += actual_energy_stored_kwh / self.vehicle_params['capacita']

                elif power < 0:  # Scarica
                    energy_delivered = -power # Potenza in kW, positiva per calcoli
                    energy_drawn_kwh = energy_delivered / self.vehicle_params['efficienza']
                    
                    # Penalità forte se si tenta di scaricare sotto soc_min_batteria
                    # Aggiunto un termine che limita la scarica effettiva per non scendere sotto il soc_min_batteria
                    max_discharge_available_kwh = (soc - self.vehicle_params['soc_min_batteria']) * self.vehicle_params['capacita']
                    actual_energy_drawn_kwh = min(energy_drawn_kwh, max_discharge_available_kwh)
                    
                    # Penalità per tentare di scaricare più del consentito
                    if energy_drawn_kwh > max_discharge_available_kwh + 1e-6: # Aggiungi una piccola tolleranza
                         total_cost += 1e6 * (energy_drawn_kwh - max_discharge_available_kwh)

                    actual_energy_delivered_kwh = actual_energy_drawn_kwh * self.vehicle_params['efficienza']
                    degradation_cost = self._calculate_degradation_cost(actual_energy_drawn_kwh)
                    revenue = horizon_prices[i] * actual_energy_delivered_kwh
                    total_cost += degradation_cost - revenue # Costo - Ricavo = Netto

                    soc -= actual_energy_drawn_kwh / self.vehicle_params['capacita']
                
                # Applica i limiti al SOC per la previsione interna
                soc = np.clip(soc, self.vehicle_params['soc_min_batteria'], self.vehicle_params['soc_max'])
                total_cost += self._calculate_anxiety_cost(soc)
            
            # Aggiungi costo/ricompensa per il SOC finale solo se è l'ultima ora dell'orizzonte e terminal
            if is_terminal:
                total_cost += self._calculate_terminal_soc_cost(soc) # Usiamo la stessa funzione che usa RL per coerenza
                
            return total_cost
        
        # Le bounds per la potenza devono essere definite per ogni ora nell'orizzonte
        bounds = [(-self.vehicle_params['p_scarica'], self.vehicle_params['p_carica'])] * n
        x0 = np.zeros(n) # Inizializza tutte le azioni a 0 (attesa)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, options={'maxiter': 200, 'ftol': 1e-7}) # Aumentato maxiter e ridotto ftol
            return res.x[0] if res.success else 0 # Restituisce solo la prima azione pianificata
        except Exception as e:
            print(f"Errore nella risoluzione MPC: {e}") # Debugging
            return 0 # In caso di errore, non fare nulla

    def train_rl_agent(self, rl_params: Dict) -> np.ndarray:
        """Addestra l'agente RL."""
        print(f"\nAddestramento agente RL in corso ({rl_params['episodes']} episodi)...")
        q_table = np.zeros((rl_params['states_ora'], rl_params['states_soc'], len(self.actions_map)))
        
        epsilon = rl_params['epsilon'] # Inizializza epsilon per il decadimento
        
        for episode in range(rl_params['episodes']):
            soc = self.sim_params['initial_soc'] # Reset SOC all'inizio di ogni episodio
            
            for ora in range(rl_params['states_ora']):
                soc_discrete = self._discretize_soc(soc, rl_params['states_soc'])
                
                # Epsilon-greedy policy
                if random.random() < epsilon:
                    azione = random.randint(0, len(self.actions_map) - 1)
                else:
                    azione = np.argmax(q_table[ora, soc_discrete])
                
                # Ottieni il risultato del passo (nuovo SOC e ricompensa)
                # Passa `last_hour` per applicare la ricompensa terminale solo alla fine
                new_soc, reward = self._get_rl_step_result(soc, ora, azione, last_hour=(ora == rl_params['states_ora'] - 1))
                
                new_soc_discrete = self._discretize_soc(new_soc, rl_params['states_soc'])
                # Per la ricompensa del prossimo stato, se non è l'ultima ora, si guarda avanti.
                # Se è l'ultima ora, il prossimo stato non ha ricompensa futura (fine episodio).
                next_q_value = 0
                if ora < rl_params['states_ora'] - 1:
                    next_q_value = np.max(q_table[ora + 1, new_soc_discrete])
                
                # Calcola Q-value
                current_q = q_table[ora, soc_discrete, azione]
                
                new_q = (1 - rl_params['alpha']) * current_q + rl_params['alpha'] * (reward + rl_params['gamma'] * next_q_value)
                q_table[ora, soc_discrete, azione] = new_q
                
                soc = new_soc # Aggiorna SOC per il passo successivo
            
            # Decadimento di epsilon
            epsilon = max(rl_params['epsilon_min'], epsilon * rl_params['epsilon_decay'])
            
            if (episode + 1) % 5000 == 0:
                print(f"Episodio {episode + 1}/{rl_params['episodes']}, Epsilon: {epsilon:.4f}")
                
        print("Addestramento RL completato!")
        # Assicurati che la directory esista prima di salvare
        q_table_dir = os.path.dirname(rl_params['q_table_file'])
        if q_table_dir and not os.path.exists(q_table_dir):
            os.makedirs(q_table_dir)
        
        np.save(rl_params['q_table_file'], q_table)
        print(f"Q-table salvata in '{rl_params['q_table_file']}'")
        return q_table

    def _get_rl_step_result(self, soc: float, ora: int, azione: int, last_hour: bool) -> Tuple[float, float]:
        """
        Calcola il risultato di un passo di RL, inclusa la ricompensa.
        La ricompensa è il guadagno economico (ricavi - costi), meno le penalità.
        """
        new_soc = soc
        reward = 0.0 # Inizializziamo la ricompensa a 0
        
        energy_cost = 0.0
        energy_revenue = 0.0
        degradation_cost = 0.0
        
        # Considera i limiti operativi della batteria prima di tutto
        if azione == 1:  # Carica
            if soc >= self.vehicle_params['soc_max']:
                reward = -5.0 # Penalità per azione invalida (tentare di caricare quando è già pieno)
            else:
                energy_purchased = self.vehicle_params['p_carica']
                energy_stored = energy_purchased * self.vehicle_params['efficienza']
                
                max_charge_kwh = (self.vehicle_params['soc_max'] - soc) * self.vehicle_params['capacita']
                actual_energy_stored = min(energy_stored, max_charge_kwh)
                actual_energy_purchased = actual_energy_stored / self.vehicle_params['efficienza'] 
                
                new_soc += actual_energy_stored / self.vehicle_params['capacita']
                degradation_cost = self._calculate_degradation_cost(actual_energy_stored)
                energy_cost = self.prices[ora] * actual_energy_purchased
                
                reward = -energy_cost - degradation_cost # Ricompensa è negativa (costo)
        
        elif azione == 2:  # Scarica
            if soc <= self.vehicle_params['soc_min_batteria']:
                reward = -5.0 # Penalità per azione invalida (tentare di scaricare quando è già vuoto)
            else:
                energy_delivered = self.vehicle_params['p_scarica']
                energy_drawn = energy_delivered / self.vehicle_params['efficienza']
                
                max_discharge_kwh = (soc - self.vehicle_params['soc_min_batteria']) * self.vehicle_params['capacita']
                actual_energy_drawn = min(energy_drawn, max_discharge_kwh)
                actual_energy_delivered = actual_energy_drawn * self.vehicle_params['efficienza'] 
                
                new_soc -= actual_energy_drawn / self.vehicle_params['capacita']
                degradation_cost = self._calculate_degradation_cost(actual_energy_drawn)
                energy_revenue = self.prices[ora] * actual_energy_delivered
                
                reward = energy_revenue - degradation_cost # Ricompensa è positiva (ricavo)
        
        # Penalità di ansia: viene sottratta perché è un costo
        anxiety_cost_step = self._calculate_anxiety_cost(new_soc)
        reward -= anxiety_cost_step

        # Aggiungi ricompensa/penalità terminale solo all'ultima ora
        if last_hour:
            terminal_cost_step = self._calculate_terminal_soc_cost(new_soc)
            reward -= terminal_cost_step # Se è un costo, lo sottraiamo dalla ricompensa; se è un ricavo, lo aggiungiamo (sottraendo un negativo)
        
        new_soc = np.clip(new_soc, 0, 1) # Assicurati che SOC rimanga tra 0 e 1
        return new_soc, reward

    def run_rl_strategy(self, q_table: np.ndarray) -> pd.DataFrame:
        """Esegue la strategia RL usando una Q-table addestrata."""
        soc = self.sim_params['initial_soc']
        log = []
        states_soc = q_table.shape[1]
        
        cumulative_costs = {'energy': 0, 'revenue': 0, 'degradation': 0, 'anxiety': 0}

        print("\nDettaglio Orario - Strategia RL:")
        print("Ora | SoC Iniziale | Prezzo | Azione | Variazione SoC | Costo Energia | Ricavo Energia | Degradazione | Ansia | Guadagno Netto")
        print("----|--------------|--------|--------|---------------|---------------|----------------|--------------|-------|---------------")
        
        for ora in range(24):
            soc_start = soc
            soc_discrete = self._discretize_soc(soc, states_soc)
            
            # Se ci sono più azioni con lo stesso valore massimo, scegli una a caso tra queste
            best_actions = np.where(q_table[ora, soc_discrete] == np.max(q_table[ora, soc_discrete]))[0]
            azione = np.random.choice(best_actions)
            action_str = self.actions_map[azione]
            
            energy_cost, energy_revenue, degradation_cost, net_hourly_gain = 0, 0, 0, 0
            
            # Recalcola i costi e i ricavi basati sull'azione scelta dall'agente
            # Questa parte dovrebbe rispecchiare la logica di _get_rl_step_result per la registrazione
            if azione == 1: # Carica
                if soc < self.vehicle_params['soc_max']:
                    energy_purchased_max = self.vehicle_params['p_carica']
                    energy_stored_max = energy_purchased_max * self.vehicle_params['efficienza']
                    
                    max_charge_kwh_available = (self.vehicle_params['soc_max'] - soc) * self.vehicle_params['capacita']
                    actual_energy_stored = min(energy_stored_max, max_charge_kwh_available)
                    actual_energy_purchased = actual_energy_stored / self.vehicle_params['efficienza']

                    soc += actual_energy_stored / self.vehicle_params['capacita']
                    degradation_cost = self._calculate_degradation_cost(actual_energy_stored)
                    energy_cost = self.prices[ora] * actual_energy_purchased
                    
            elif azione == 2: # Scarica
                if soc > self.vehicle_params['soc_min_batteria']:
                    energy_delivered_max = self.vehicle_params['p_scarica']
                    energy_drawn_max = energy_delivered_max / self.vehicle_params['efficienza']
                    
                    max_discharge_kwh_available = (soc - self.vehicle_params['soc_min_batteria']) * self.vehicle_params['capacita']
                    actual_energy_drawn = min(energy_drawn_max, max_discharge_kwh_available)
                    actual_energy_delivered = actual_energy_drawn * self.vehicle_params['efficienza']

                    soc -= actual_energy_drawn / self.vehicle_params['capacita']
                    degradation_cost = self._calculate_degradation_cost(actual_energy_drawn)
                    energy_revenue = self.prices[ora] * actual_energy_delivered
            
            anxiety_cost = self._calculate_anxiety_cost(soc)
            net_hourly_gain = energy_revenue - energy_cost - degradation_cost - anxiety_cost

            cumulative_costs['energy'] += energy_cost
            cumulative_costs['revenue'] += energy_revenue
            cumulative_costs['degradation'] += degradation_cost
            cumulative_costs['anxiety'] += anxiety_cost
            
            soc = np.clip(soc, 0, 1)
            
            self._log_hourly_data(log, ora, soc_start, self.prices[ora], action_str, energy_cost,
                                  energy_revenue, degradation_cost, anxiety_cost, net_hourly_gain, 'Guadagno Netto Ora (€)')

        # Alla fine della giornata, aggiungi la componente terminale al costo totale
        final_soc_cost_summary = self._calculate_terminal_soc_cost(soc)
        cumulative_costs['anxiety'] += final_soc_cost_summary # Aggiunto al totale per coerenza
        
        return self._finalize_log(log, cumulative_costs, 'Guadagno Netto Ora (€)')

    def _discretize_soc(self, soc: float, states: int) -> int:
        """Discretizza il SOC in bin di stato."""
        # Converti SOC da [0,1] a indice intero [0, states-1]
        return int(np.clip(round(soc * (states - 1)), 0, states - 1))

    def _finalize_log(self, log: List, cumulative_costs: Dict, reward_col_name: str) -> pd.DataFrame:
        """Aggiunge il riepilogo al log e lo converte in DataFrame."""
        total_gain = cumulative_costs['revenue'] - cumulative_costs['energy'] - cumulative_costs['degradation'] - cumulative_costs['anxiety']
        
        print("\nRIEPILOGO GIORNALIERO:")
        print(f"  - Costo Energia Totale: {cumulative_costs['energy']:.4f} €")
        print(f"  - Ricavo Energia Totale: {cumulative_costs['revenue']:.4f} €")
        print(f"  - Costo Degradazione Totale: {cumulative_costs['degradation']:.4f} €")
        print(f"  - Costo Ansia Totale: {cumulative_costs['anxiety']:.4f} €")
        print(f"  - GUADAGNO NETTO TOTALE: {total_gain:.4f} €")
        
        # Aggiungi riga di riepilogo
        summary_row = {col: '' for col in log[0].keys()} # Inizializza con colonne vuote
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

def compare_strategies(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Confronta le performance delle diverse strategie.

    Args:
        results (Dict[str, pd.DataFrame]): Dizionario con i risultati di ogni strategia.

    Returns:
        pd.DataFrame: DataFrame di confronto.
    """
    print("\n" + "=" * 80)
    print("CONFRONTO PRESTAZIONI STRATEGIE")
    print("=" * 80)
    
    comparison = []
    for strategy, df in results.items():
        summary_row = df.iloc[-1]
        reward_col = 'Guadagno Netto Ora (€)' if 'Guadagno Netto Ora (€)' in df.columns else 'Ricompensa (€)'
        
        # Recupera il SOC finale dall'ultima riga di operazione, non dalla riga "TOTALE"
        if len(df) > 1 and df.iloc[-1]['Ora'] == 'TOTALE':
            last_op_row = df.iloc[-2] # Prendiamo la riga prima dell'ultima (che è il totale)
        else:
            last_op_row = df.iloc[-1] # Se non c'è una riga totale, prendiamo l'ultima
        
        soc_initial_for_final = last_op_row.get('SoC Iniziale (%)', np.nan)
        soc_variation = last_op_row.get('Variazione SoC (%)', 0)
        
        # Calcola il SOC finale in base al SOC iniziale dell'ultima ora e la variazione di quella stessa ora
        soc_final = soc_initial_for_final / 100 + soc_variation / 100 if not np.isnan(soc_initial_for_final) else np.nan
        soc_final = np.clip(soc_final, 0, 1) * 100 # Assicurati che sia tra 0 e 100%

        comparison.append({
            'Strategia': strategy,
            'Guadagno Netto (€)': summary_row[reward_col],
            'SOC Finale (%)': round(soc_final, 1) if not np.isnan(soc_final) else 'N/A',
            'Costo Degradazione (€)': summary_row['Costo Degradazione (€)'],
            'Costo Ansia (€)': summary_row['Costo Ansia (€)']
        })
    
    return pd.DataFrame(comparison)

def load_energy_prices_from_file(file_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Carica i prezzi dell'energia da un file Excel.

    Args:
        file_path (str): Percorso del file Excel.

    Returns:
        Tuple[pd.DataFrame, List[str]]: DataFrame dei prezzi e lista delle zone disponibili.
    """
    try:
        df = pd.read_excel(file_path)
        print(f"File '{file_path}' caricato con successo.")
        if 'Ora' not in df.columns:
            raise ValueError("Il file Excel deve contenere una colonna 'Ora'.")
        
        available_zones = []
        for col in df.columns:
            if col == 'Ora':
                continue
            if df[col].dtype == 'object':
                try:
                    df[col] = df[col].str.replace(',', '.', regex=False).astype(float)
                    available_zones.append(col)
                except (ValueError, AttributeError):
                    print(f"Attenzione: la colonna '{col}' non è numerica e sarà ignorata.")
            else:
                available_zones.append(col)
        return df, available_zones
    except FileNotFoundError:
        print(f"ERRORE: Il file '{file_path}' non è stato trovato.")
        return None, None
    except Exception as e:
        print(f"ERRORE: Impossibile leggere il file Excel. Dettagli: {e}")
        return None, None

def explain_anxiety_cost():
    """Stampa una spiegazione del concetto di 'costo ansia'."""
    print("\n" + "=" * 80)
    print("SPIEGAZIONE COSTO ANSIA (RANGE ANXIETY)")
    print("=" * 80)
    print("Il 'costo ansia' modella il disagio dell'utente quando la carica (SOC)")
    print("scende sotto una soglia minima desiderata, rappresentando economicamente")
    print("il valore della 'tranquillità' di avere una riserva di energia.")
    print("Questo costo può influenzare pesantemente le decisioni degli algoritmi,")
    print("soprattutto se impostato troppo alto, spingendo il sistema a mantenere")
    print("un SOC elevato anche a costo di minori guadagni economici.")
    print("La calibrazione di questo parametro è cruciale per bilanciare")
    print("l'ottimizzazione economica con la soddisfazione dell'utente.")
    print("\nIn questa versione, è stata aggiunta anche una ricompensa/penalità per")
    print("il SOC finale, per incentivare l'agente RL a terminare la giornata")
    print("con un livello di carica desiderato, bilanciando così la paura di scaricare")
    print("e l'opportunità di massimizzare i profitti.")
    print("=" * 80)

def get_user_config(sim_params: Dict) -> Dict:
    """Ottiene i parametri di simulazione personalizzati dall'utente (versione non interattiva)."""
    print("\n" + "=" * 80)
    print("UTILIZZO PARAMETRI DI DEFAULT PER SIMULAZIONE")
    print("=" * 80)
    # Assegna direttamente i valori di default per l'esecuzione non interattiva
    return sim_params

def load_price_data(file_path: str = None, zone_name: str = None) -> Dict[int, float]:
    """Carica i dati dei prezzi (versione non interattiva)."""
    print("\n" + "=" * 80)
    print("CARICAMENTO DATI PREZZI (NON INTERATTIVO)")
    print("=" * 80)
    
    # Usa il percorso predefinito e la zona 'Italia' per l'esecuzione non interattiva
    # Assicurati che il file 'PrezziZonali.xlsx' sia nella sottocartella 'downloads'
    script_dir = os.path.dirname(__file__) if '__file__' in locals() else os.getcwd()
    default_path = os.path.join(script_dir, "downloads", "PrezziZonali.xlsx")
    
    file_path = file_path if file_path else default_path
    zona_scelta = zone_name if zone_name else "Italia"
    
    df_prezzi, zone_disponibili = load_energy_prices_from_file(file_path)
    if df_prezzi is None:
        sys.exit("Impossibile caricare i prezzi dell'energia. Uscita.")
        
    if zona_scelta not in zone_disponibili:
        print(f"ERRORE: Zona '{zona_scelta}' non trovata nel file prezzi. Zone disponibili: {zone_disponibili}")
        sys.exit("Zona non valida. Uscita.")
    
    print(f"\nHai selezionato la zona: '{zona_scelta}'")
    
    # Restituisce un dizionario ora -> prezzo in €/kWh (dal momento che il file è in €/MWh)
    return {int(row['Ora']) - 1: row[zona_scelta] / 1000 for _, row in df_prezzi.iterrows()}

def run_and_compare_strategies(optimizer: V2GOptimizer, rl_params: Dict) -> Dict:
    """Esegue tutte le strategie e restituisce i risultati."""
    print("\n" + "=" * 80)
    print("ESECUZIONE STRATEGIE")
    print("=" * 80)
    
    print("\n>>> STRATEGIA EURISTICA SEMPLICE <<<")
    heuristic_df = optimizer.run_heuristic_strategy()

    print("\n>>> STRATEGIA EURISTICA LCVF <<<")
    lcvf_df = optimizer.run_lcvf_strategy()
    
    print("\n>>> STRATEGIA MPC (ORIZZONTE CORTO) <<<")
    mpc_short_horizon_df = optimizer.run_mpc_strategy(horizon=SIMULATION_PARAMS['mpc_horizon'])

    print("\n>>> STRATEGIA MPC (ORIZZONTE 24 ORE) <<<")
    mpc_full_horizon_df = optimizer.run_mpc_strategy(horizon=24)
    
    print("\n>>> STRATEGIA RL <<<")
    
    # Crea la directory per la Q-table se non esiste
    q_table_dir = os.path.dirname(rl_params['q_table_file'])
    if q_table_dir and not os.path.exists(q_table_dir):
        os.makedirs(q_table_dir)

    if os.path.exists(rl_params['q_table_file']):
        print(f"Caricamento Q-table da '{rl_params['q_table_file']}'...")
        q_table = np.load(rl_params['q_table_file'])
    else:
        q_table = optimizer.train_rl_agent(rl_params)
    rl_df = optimizer.run_rl_strategy(q_table)
    
    return {
        'Euristica Semplice': heuristic_df,
        'Euristica LCVF': lcvf_df,
        f"MPC (O={SIMULATION_PARAMS['mpc_horizon']}h)": mpc_short_horizon_df,
        'MPC (O=24h)': mpc_full_horizon_df,
        'Reinforcement Learning': rl_df
    }

def save_results_to_excel(results: Dict, output_dir: str):
    """Salva i risultati e il confronto in un file Excel."""
    output_path = os.path.join(output_dir, "Risultati_V2G_ottimizzati_v2.xlsx") # Nome file cambiato per la nuova versione
    
    # Assicurati che la directory di output esista
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with pd.ExcelWriter(output_path) as writer:
        for name, df in results.items():
            df.to_excel(writer, sheet_name=name, index=False)
        
        comparison_df = compare_strategies(results)
        comparison_df.to_excel(writer, sheet_name='Confronto', index=False)
        
        print("\n" + "=" * 80)
        print("RIEPILOGO FINALE")
        print("=" * 80)
        print(comparison_df.to_string(index=False))

    print(f"\nRisultati salvati in: {output_path}")

# ========================================================================
# ESECUZIONE PRINCIPALE
# ========================================================================

def main():
    """Funzione principale per eseguire il simulatore V2G."""
    explain_anxiety_cost()
    
    # Assicurati che sim_config sia un copy di SIMULATION_PARAMS per evitare modifiche dirette
    sim_config = get_user_config(SIMULATION_PARAMS.copy()) 
    
    prezzi_per_zona = load_price_data()
    
    optimizer = V2GOptimizer(VEHICLE_PARAMS, sim_config, prezzi_per_zona)
    
    results = run_and_compare_strategies(optimizer, RL_PARAMS)
    
    # La directory per il file Q-table dovrebbe essere la stessa per i risultati
    output_dir = os.path.dirname(RL_PARAMS['q_table_file'])
    if not output_dir: 
        output_dir = '.' # Se RL_PARAMS['q_table_file'] è solo un nome file, usa la directory corrente
    
    save_results_to_excel(results, output_dir)
    
    # Stampa del guadagno teorico massimo
    max_price = max(prezzi_per_zona.values())
    min_price = min(prezzi_per_zona.values())
    
    # Calcola il range operativo utilizzabile dalla batteria per V2G
    usable_capacity_kwh = VEHICLE_PARAMS['capacita'] * (VEHICLE_PARAMS['soc_max'] - VEHICLE_PARAMS['soc_min_batteria'])
    
    # Guadagno teorico massimo: comprare tutta la capacità utilizzabile al prezzo minimo e venderla al prezzo massimo
    # Considera l'efficienza bidirezionale per un ciclo completo di carica e scarica
    max_theoretical_gain = usable_capacity_kwh * (max_price * VEHICLE_PARAMS['efficienza'] - min_price / VEHICLE_PARAMS['efficienza'])
    
    print("\n" + "=" * 80)
    print("MASSIMO GUADAGNO TEORICO")
    print("=" * 80)
    print(f"  - Capacità utilizzabile per V2G: {usable_capacity_kwh:.2f} kWh (dal {VEHICLE_PARAMS['soc_min_batteria']*100}% al {VEHICLE_PARAMS['soc_max']*100}%)")
    print(f"  - Prezzo minimo giornaliero: {min_price:.4f} €/kWh")
    print(f"  - Prezzo massimo giornaliero: {max_price:.4f} €/kWh")
    print(f"  - Guadagno teorico massimo (considerando efficienza bidirezionale): €{max_theoretical_gain:.2f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
