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
    'capacita': 60,           # kWh
    'p_carica': 7.4,          # kW
    'p_scarica': 5.0,         # kW
    'efficienza': 0.92,       #
    'soc_max': 0.9,           # 90%
    'soc_min_batteria': 0.1,  # 10%
    'costo_degradazione': 0.02, # €/kWh
}

SIMULATION_PARAMS = {
    'soc_min_utente': 0.3,    # 30%
    'penalita_ansia': 0.15,   # €/%
    'initial_soc': 0.5,       # 50%
    'mpc_horizon': 6,         # ore
}

RL_PARAMS = {
    'states_ora': 24,
    'states_soc': 11,
    'alpha': 0.1,
    'gamma': 0.95,
    'epsilon': 0.1,
    'episodes': 20000, # Aumentato gli episodi di addestramento a 20000
    'q_table_file': 'q_table.npy'
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
        self.anxiety_penalty = self.sim_params['penalita_ansia']

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
        soc_min_user = self.sim_params['soc_min_utente']
        if soc < soc_min_user:
            deficit = soc_min_user - soc
            return self.anxiety_penalty * deficit * 100
        return 0.0

    def _calculate_terminal_soc_cost(self, soc: float) -> float:
        """Calcola il costo se il SOC finale è inferiore al target."""
        target_soc = self.sim_params.get('soc_target_finale', 0.5)
        if soc < target_soc:
            deficit = target_soc - soc
            # Penalità simile a quella dell'ansia ma per lo stato finale
            return self.anxiety_penalty * 1.5 * deficit * 100
        return 0.0

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
            'Variazione SoC (%)': variation_soc,
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
        
        charge_threshold = avg_price - 0.5 * std_price
        discharge_threshold = avg_price + 0.5 * std_price
        
        cumulative_costs = {'energy': 0, 'revenue': 0, 'degradation': 0, 'anxiety': 0}

        print("\nDettaglio Orario - Strategia Euristica:")
        print("Ora | SoC Iniziale | Prezzo | Azione | Variazione SoC | Costo Energia | Ricavo Energia | Degradazione | Ansia | Guadagno Netto")
        print("----|--------------|--------|--------|---------------|---------------|----------------|--------------|-------|---------------")
        
        for hour, price in self.prices.items():
            soc_start = soc
            action, energy_cost, energy_revenue, degradation_cost = 'Attesa', 0, 0, 0

            if price < charge_threshold and soc < self.vehicle_params['soc_max']:
                action = 'Carica'
                energy_purchased = self.vehicle_params['p_carica']
                energy_stored = energy_purchased * self.vehicle_params['efficienza']
                soc += energy_stored / self.vehicle_params['capacita']
                degradation_cost = self._calculate_degradation_cost(energy_stored)
                energy_cost = price * energy_purchased
                
            elif price > discharge_threshold and soc > self.vehicle_params['soc_min_batteria']:
                action = 'Scarica'
                energy_delivered = self.vehicle_params['p_scarica']
                energy_drawn = energy_delivered / self.vehicle_params['efficienza']
                soc -= energy_drawn / self.vehicle_params['capacita']
                degradation_cost = self._calculate_degradation_cost(energy_drawn)
                energy_revenue = price * energy_delivered
            
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

    def run_lcvf_strategy(self) -> pd.DataFrame:
        """Esegue la strategia LCVF (Load Conservation Valley-Filling) con V2G."""
        soc = self.sim_params['initial_soc']
        log = []
        cumulative_costs = {'energy': 0, 'revenue': 0, 'degradation': 0, 'anxiety': 0}

        # --- Fase di Pianificazione LCVF con V2G ---
        # Identifica le ore migliori per caricare e scaricare durante la giornata
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
                energy_purchased = self.vehicle_params['p_carica']
                energy_stored = energy_purchased * self.vehicle_params['efficienza']
                soc += energy_stored / self.vehicle_params['capacita']
                degradation_cost = self._calculate_degradation_cost(energy_stored)
                energy_cost = price * energy_purchased
            
            elif planned_action == 'Scarica' and soc > self.vehicle_params['soc_min_batteria']:
                action = 'Scarica'
                energy_delivered = self.vehicle_params['p_carica'] # Scarica alla stessa potenza di carica per semplicità
                energy_drawn = energy_delivered / self.vehicle_params['efficienza']
                soc -= energy_drawn / self.vehicle_params['capacita']
                degradation_cost = self._calculate_degradation_cost(energy_drawn)
                energy_revenue = price * energy_delivered

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
        
        # Il costo per il SOC finale si applica solo se l'orizzonte è l'intera giornata
        apply_terminal_cost = (horizon >= len(hours))

        cumulative_costs = {'energy': 0, 'revenue': 0, 'degradation': 0, 'anxiety': 0}

        print(f"\nDettaglio Orario - Strategia MPC (Orizzonte={horizon}h):")
        print("Ora | SoC Iniziale | Prezzo | Azione | Variazione SoC | Costo Energia | Ricavo Energia | Degradazione | Ansia | Guadagno Netto")
        print("----|--------------|--------|--------|---------------|---------------|----------------|--------------|-------|---------------")
        
        for i, hour in enumerate(hours):
            soc_start = soc
            action, energy_cost, energy_revenue, degradation_cost = 'Attesa', 0, 0, 0
            
            horizon_hours = hours[i:i+horizon]
            if len(horizon_hours) >= 2:
                horizon_prices = [self.prices[h] for h in horizon_hours]
                action_power = self._solve_mpc(soc, horizon_prices, apply_terminal_cost)
                
                if action_power > 0.1:
                    action = 'Carica'
                    power = min(action_power, self.vehicle_params['p_carica'])
                    energy_stored = power * self.vehicle_params['efficienza']
                    soc += energy_stored / self.vehicle_params['capacita']
                    degradation_cost = self._calculate_degradation_cost(energy_stored)
                    energy_cost = self.prices[hour] * power
                    
                elif action_power < -0.1:
                    action = 'Scarica'
                    power = min(-action_power, self.vehicle_params['p_scarica'])
                    energy_drawn = power / self.vehicle_params['efficienza']
                    soc -= energy_drawn / self.vehicle_params['capacita']
                    degradation_cost = self._calculate_degradation_cost(energy_drawn)
                    energy_revenue = self.prices[hour] * power
            
            anxiety_cost = self._calculate_anxiety_cost(soc)
            net_gain = energy_revenue - energy_cost - degradation_cost - anxiety_cost
            
            cumulative_costs['energy'] += energy_cost
            cumulative_costs['revenue'] += energy_revenue
            cumulative_costs['degradation'] += degradation_cost
            cumulative_costs['anxiety'] += anxiety_cost
            
            soc = np.clip(soc, 0, 1)
            
            self._log_hourly_data(log, hour, soc_start, self.prices[hour], action, energy_cost, 
                                  energy_revenue, degradation_cost, anxiety_cost, net_gain, 'Guadagno Netto Ora (€)')

        return self._finalize_log(log, cumulative_costs, 'Guadagno Netto Ora (€)')

    def _solve_mpc(self, current_soc: float, horizon_prices: List[float], is_terminal: bool) -> float:
        """Risolve il problema di ottimizzazione MPC per l'orizzonte dato."""
        n = len(horizon_prices)
        
        def objective(x):
            total_cost = 0
            soc = current_soc
            final_soc = soc
            
            for i in range(n):
                power = x[i]
                cost_i = 0
                
                if power > 0:  # Carica
                    energy_stored = power * self.vehicle_params['efficienza']
                    degradation_cost = self._calculate_degradation_cost(energy_stored)
                    cost_i = horizon_prices[i] * power + degradation_cost
                    soc += energy_stored / self.vehicle_params['capacita']
                elif power < 0:  # Scarica
                    energy_drawn = -power / self.vehicle_params['efficienza']
                    degradation_cost = self._calculate_degradation_cost(energy_drawn)
                    revenue = horizon_prices[i] * -power
                    cost_i = degradation_cost - revenue
                    soc -= energy_drawn / self.vehicle_params['capacita']
                
                cost_i += self._calculate_anxiety_cost(soc)
                total_cost += cost_i
                final_soc = np.clip(soc, self.vehicle_params['soc_min_batteria'], self.vehicle_params['soc_max'])

            # Aggiungi costo per il SOC finale solo se specificato
            if is_terminal:
                total_cost += self._calculate_terminal_soc_cost(final_soc)
                
            return total_cost
        
        bounds = [(-self.vehicle_params['p_scarica'], self.vehicle_params['p_carica'])] * n
        x0 = np.zeros(n)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, options={'maxiter': 50, 'ftol': 1e-4})
            return res.x[0] if res.success else 0
        except Exception:
            return 0

    def train_rl_agent(self, rl_params: Dict) -> np.ndarray:
        """Addestra l'agente RL."""
        print(f"\nAddestramento agente RL in corso ({rl_params['episodes']} episodi)...")
        q_table = np.zeros((rl_params['states_ora'], rl_params['states_soc'], len(self.actions_map)))
        
        for _ in range(rl_params['episodes']):
            soc = self.sim_params['initial_soc']
            for ora in range(rl_params['states_ora']):
                soc_discrete = self._discretize_soc(soc, rl_params['states_soc'])
                
                if random.random() < rl_params['epsilon']:
                    azione = random.randint(0, len(self.actions_map) - 1)
                else:
                    azione = np.argmax(q_table[ora, soc_discrete])
                
                new_soc, reward = self._get_rl_step_result(soc, ora, azione)
                
                new_soc_discrete = self._discretize_soc(new_soc, rl_params['states_soc'])
                next_ora = (ora + 1) % rl_params['states_ora']
                
                best_next_q = np.max(q_table[next_ora, new_soc_discrete])
                current_q = q_table[ora, soc_discrete, azione]
                new_q = (1 - rl_params['alpha']) * current_q + rl_params['alpha'] * (reward + rl_params['gamma'] * best_next_q)
                q_table[ora, soc_discrete, azione] = new_q
                
                soc = new_soc
        
        print("Addestramento RL completato!")
        np.save(rl_params['q_table_file'], q_table)
        print(f"Q-table salvata in '{rl_params['q_table_file']}'")
        return q_table

    def _get_rl_step_result(self, soc: float, ora: int, azione: int) -> Tuple[float, float]:
        """Calcola il risultato di un passo di RL."""
        new_soc = soc
        reward = 0
        
        if azione == 1 and soc < self.vehicle_params['soc_max']:  # Carica
            energy_purchased = self.vehicle_params['p_carica']
            energy_stored = energy_purchased * self.vehicle_params['efficienza']
            new_soc += energy_stored / self.vehicle_params['capacita']
            degradation_cost = self._calculate_degradation_cost(energy_stored)
            reward = -self.prices[ora] * energy_purchased - degradation_cost
            
        elif azione == 2 and soc > self.vehicle_params['soc_min_batteria']:  # Scarica
            energy_delivered = self.vehicle_params['p_scarica']
            energy_drawn = energy_delivered / self.vehicle_params['efficienza']
            new_soc -= energy_drawn / self.vehicle_params['capacita']
            degradation_cost = self._calculate_degradation_cost(energy_drawn)
            reward = self.prices[ora] * energy_delivered - degradation_cost
        
        reward -= self._calculate_anxiety_cost(new_soc)
        new_soc = np.clip(new_soc, 0, 1)
        return new_soc, reward

    def run_rl_strategy(self, q_table: np.ndarray) -> pd.DataFrame:
        """Esegue la strategia RL usando una Q-table addestrata."""
        soc = self.sim_params['initial_soc']
        log = []
        states_soc = q_table.shape[1]
        
        cumulative_costs = {'energy': 0, 'revenue': 0, 'degradation': 0, 'anxiety': 0}

        print("\nDettaglio Orario - Strategia RL:")
        print("Ora | SoC Iniziale | Prezzo | Azione | Variazione SoC | Costo Energia | Ricavo Energia | Degradazione | Ansia | Ricompensa")
        print("----|--------------|--------|--------|---------------|---------------|----------------|--------------|-------|-----------")
        
        for ora in range(24):
            soc_start = soc
            soc_discrete = self._discretize_soc(soc, states_soc)
            azione = np.argmax(q_table[ora, soc_discrete])
            action_str = self.actions_map[azione]
            
            energy_cost, energy_revenue, degradation_cost, net_reward = 0, 0, 0, 0
            
            if azione == 1 and soc < self.vehicle_params['soc_max']:
                energy_purchased = self.vehicle_params['p_carica']
                energy_stored = energy_purchased * self.vehicle_params['efficienza']
                soc += energy_stored / self.vehicle_params['capacita']
                degradation_cost = self._calculate_degradation_cost(energy_stored)
                energy_cost = self.prices[ora] * energy_purchased
                net_reward = -energy_cost - degradation_cost
                
            elif azione == 2 and soc > self.vehicle_params['soc_min_batteria']:
                energy_delivered = self.vehicle_params['p_scarica']
                energy_drawn = energy_delivered / self.vehicle_params['efficienza']
                soc -= energy_drawn / self.vehicle_params['capacita']
                degradation_cost = self._calculate_degradation_cost(energy_drawn)
                energy_revenue = self.prices[ora] * energy_delivered
                net_reward = energy_revenue - degradation_cost
            
            anxiety_cost = self._calculate_anxiety_cost(soc)
            net_reward -= anxiety_cost
            
            cumulative_costs['energy'] += energy_cost
            cumulative_costs['revenue'] += energy_revenue
            cumulative_costs['degradation'] += degradation_cost
            cumulative_costs['anxiety'] += anxiety_cost
            
            soc = np.clip(soc, 0, 1)
            
            self._log_hourly_data(log, ora, soc_start, self.prices[ora], action_str, energy_cost,
                                  energy_revenue, degradation_cost, anxiety_cost, net_reward, 'Ricompensa (€)')

        return self._finalize_log(log, cumulative_costs, 'Ricompensa (€)')

    def _discretize_soc(self, soc: float, states: int) -> int:
        """Discretizza il SOC in bin di stato."""
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
        
        log.append({
            'Ora': 'TOTALE',
            'Costo Energia (€)': round(cumulative_costs['energy'], 4),
            'Ricavo Energia (€)': round(cumulative_costs['revenue'], 4),
            'Costo Degradazione (€)': round(cumulative_costs['degradation'], 4),
            'Costo Ansia (€)': round(cumulative_costs['anxiety'], 4),
            reward_col_name: round(total_gain, 4)
        })
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
        
        last_op_row = df.iloc[-2]
        soc_final = last_op_row['SoC Iniziale (%)'] + last_op_row['Variazione SoC (%)']
        
        comparison.append({
            'Strategia': strategy,
            'Guadagno Netto (€)': summary_row[reward_col],
            'SOC Finale (%)': soc_final,
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
    print("=" * 80)

def get_user_config(sim_params: Dict) -> Dict:
    """Ottiene i parametri di simulazione personalizzati dall'utente."""
    print("\n" + "=" * 80)
    print("PERSONALIZZAZIONE PARAMETRI PSICOLOGICI")
    print("=" * 80)
    sim_params['soc_min_utente'] = float(input(f"Inserisci soglia ansia SOC (default {sim_params['soc_min_utente']}): ") or sim_params['soc_min_utente'])
    VEHICLE_PARAMS['soc_max'] = float(input(f"Inserisci SOC massimo (default {VEHICLE_PARAMS['soc_max']}): ") or VEHICLE_PARAMS['soc_max'])
    VEHICLE_PARAMS['soc_min_batteria'] = float(input(f"Inserisci SOC minimo fisico (default {VEHICLE_PARAMS['soc_min_batteria']}): ") or VEHICLE_PARAMS['soc_min_batteria'])
    VEHICLE_PARAMS['costo_degradazione'] = float(input(f"Costo degradazione (€/kWh, default {VEHICLE_PARAMS['costo_degradazione']}): ") or VEHICLE_PARAMS['costo_degradazione'])
    sim_params['penalita_ansia'] = float(input(f"Penalità ansia (€/%, default {sim_params['penalita_ansia']}): ") or sim_params['penalita_ansia'])
    sim_params['soc_target_finale'] = float(input(f"SOC target finale (default {sim_params.get('soc_target_finale', 0.5)}): ") or sim_params.get('soc_target_finale', 0.5))
    return sim_params

def load_price_data(file_path: str = None, zone_name: str = None) -> Dict[int, float]:
    """Carica i dati dei prezzi e permette all'utente di scegliere una zona."""
    print("\n" + "=" * 80)
    print("CARICAMENTO DATI PREZZI")
    print("=" * 80)
    
    if file_path is None:
        default_path = os.path.join(os.path.dirname(__file__), "downloads", "PrezziZonali.xlsx")
        file_path = input(f"Percorso file Excel [default: '{default_path}']: ").strip() or default_path
    
    df_prezzi, zone_disponibili = load_energy_prices_from_file(file_path)
    if df_prezzi is None:
        sys.exit()
        
    if zone_name:
        zona_scelta = zone_name
        if zona_scelta not in zone_disponibili:
            print(f"ERRORE: Zona '{zone_name}' non trovata nel file prezzi. Zone disponibili: {zone_disponibili}")
            sys.exit()
    else:
        print("\nZone disponibili per l'analisi:")
        for i, zona in enumerate(zone_disponibili, 1):
            print(f"{i}. {zona}")
        
        zona_scelta = None
        while zona_scelta not in zone_disponibili:
            scelta = input("\nInserisci il numero o il nome della zona: ").strip()
            try:
                idx = int(scelta) - 1
                if 0 <= idx < len(zone_disponibili):
                    zona_scelta = zone_disponibili[idx]
            except ValueError:
                if scelta in zone_disponibili:
                    zona_scelta = scelta
    
    print(f"\nHai selezionato la zona: '{zona_scelta}'")
    
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
    output_path = os.path.join(output_dir, "Risultati_V2G.xlsx")
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
    
    sim_config = get_user_config(SIMULATION_PARAMS.copy())
    
    prezzi_per_zona = load_price_data()
    
    optimizer = V2GOptimizer(VEHICLE_PARAMS, sim_config, prezzi_per_zona)
    
    results = run_and_compare_strategies(optimizer, RL_PARAMS)
    
    output_dir = os.path.dirname(RL_PARAMS['q_table_file']) or '.'
    save_results_to_excel(results, output_dir)
    
    # Stampa del guadagno teorico massimo
    max_price = max(prezzi_per_zona.values())
    min_price = min(prezzi_per_zona.values())
    usable_capacity = VEHICLE_PARAMS['capacita'] * (VEHICLE_PARAMS['soc_max'] - sim_config['soc_min_utente'])
    max_theoretical_gain = usable_capacity * (max_price - min_price) * VEHICLE_PARAMS['efficienza']**2
    
    print("\n" + "=" * 80)
    print("MASSIMO GUADAGNO TEORICO")
    print("=" * 80)
    print(f"  - Capacità utilizzabile: {usable_capacity:.2f} kWh")
    print(f"  - Prezzo minimo: {min_price:.4f} €/kWh")
    print(f"  - Prezzo massimo: {max_price:.4f} €/kWh")
    print(f"  - Guadagno teorico (considerando efficienza): €{max_theoretical_gain:.2f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
