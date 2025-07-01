# -*- coding: utf-8 -*-

# =======================================================================
# SCRIPT PER ANALISI DI SENSIBILITÀ V2G
# Author: Angelo Caravella
# Version: 2.0 
# Description: Questo script esegue un'analisi di sensibilità completa,
#              iterando su diverse combinazioni di parametri per valutare
#              la robustezza delle strategie V2G implementate in Mpc.py.
#              I risultati vengono aggregati per calcolare le performance
#              medie e salvati in un file Excel.
# =======================================================================

import numpy as np
import pandas as pd
import itertools
import os
import sys
from tqdm import tqdm

# =======================================================================
# FASE 1: IMPORTAZIONE E CONTROLLO
# =======================================================================
try:
    # Importa le classi e le funzioni necessarie dal tuo script principale
    from New import (V2GOptimizer, load_price_data, split_data, 
                     compare_strategies, save_results_to_excel,
                     RL_PARAMS, SIMULATION_PARAMS, VEHICLE_PARAMS)
except ImportError:
    print("ERRORE: Assicurati che il file 'Mpc.py' sia nella stessa cartella.")
    sys.exit()

# =======================================================================
# FASE 2: DEFINIZIONE DEI PARAMETRI PER L'ANALISI DI SENSIBILITÀ
# =======================================================================
# Definiamo qui i range di valori da testare.
# NOTA: Il 'costo_degradazione' è stato sostituito con 'costo_batteria'
# per essere compatibile con i nuovi modelli di degradazione.
PARAM_RANGES = {
    'soc_min_utente': [0.2, 0.4, 0.6],       # Da utente "rischioso" a prudente
    'penalita_ansia': [0.005, 0.01, 0.02],   # Da penalità leggera a severa (€/%)
    'costo_batteria': [120 * 60, 150 * 60, 200 * 60], # Costo totale batteria (es. 120, 150, 200 €/kWh)
    'soc_target_finale': [0.5, 0.7, 0.85]      # Obiettivi di fine giornata
}

# =======================================================================
# FASE 3: FUNZIONE PRINCIPALE PER L'ANALISI
# =======================================================================
def run_sensitivity_analysis():
    """
    Esegue un'analisi di sensibilità completa, iterando su tutte le
    combinazioni dei parametri definiti in PARAM_RANGES.
    """
    print("=" * 80)
    print("AVVIO ANALISI DI SENSIBILITA' SULLE STRATEGIE V2G")
    print("=" * 80)

    # --- 1. Caricamento e Divisione Dati (una sola volta) ---
    all_price_data = load_price_data()
    training_profiles, test_profile = split_data(all_price_data, test_zone="Italia")

    # --- 2. Addestramento o Caricamento Agente RL (una sola volta) ---
    # L'agente viene addestrato sul training set per generalizzare.
    temp_optimizer_for_training = V2GOptimizer(VEHICLE_PARAMS, SIMULATION_PARAMS)
    q_table_file = RL_PARAMS['q_table_file']
    
    if os.path.exists(q_table_file):
        print(f"\nCaricamento Q-table esistente da '{q_table_file}'...")
        q_table = np.load(q_table_file)
    else:
        print(f"\nNessuna Q-table trovata. Avvio addestramento una tantum sul training set...")
        q_table = temp_optimizer_for_training.train_rl_agent(RL_PARAMS, training_profiles)
    
    # --- 3. Creazione delle combinazioni di parametri ---
    keys, values = zip(*PARAM_RANGES.items())
    parameter_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"\nVerranno eseguite {len(parameter_combinations)} simulazioni per ogni strategia sulla giornata di test.")
    
    all_run_results = []

    # --- 4. Esecuzione del ciclo di simulazioni ---
    for params in tqdm(parameter_combinations, desc="Analisi di sensibilità"):
        
        # Crea una configurazione specifica per questa iterazione
        sim_config = SIMULATION_PARAMS.copy()
        sim_config['soc_min_utente'] = params['soc_min_utente']
        sim_config['penalita_ansia'] = params['penalita_ansia']
        sim_config['soc_target_finale'] = params['soc_target_finale']

        vehicle_config = VEHICLE_PARAMS.copy()
        vehicle_config['costo_batteria'] = params['costo_batteria']

        # Inizializza l'ottimizzatore con i parametri correnti
        optimizer = V2GOptimizer(vehicle_config, sim_config)
        # Imposta la giornata di test per la valutazione
        optimizer.set_prices_for_simulation(test_profile)

        # Definisci le strategie da eseguire
        strategies = {
            "Euristica Semplice": optimizer.run_heuristic_strategy,
            "Euristica LCVF": optimizer.run_lcvf_strategy,
            f"MPC (O={sim_config['mpc_horizon']}h)": lambda: optimizer.run_mpc_strategy(horizon=sim_config['mpc_horizon']),
            "Reinforcement Learning": lambda: optimizer.run_rl_strategy(q_table)
        }

        for name, func in strategies.items():
            # Sopprimiamo la stampa dettagliata per pulire l'output
            original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            try:
                df_result = func()
            finally:
                sys.stdout.close()
                sys.stdout = original_stdout

            if df_result.empty: continue

            # Estrai la riga di riepilogo
            summary = df_result.iloc[-1].to_dict()
            summary.update(params)
            summary['Strategia'] = name
            
            # Standardizza il nome della colonna del guadagno
            if 'Guadagno Netto Ora (€)' in summary:
                summary['Guadagno Netto (€)'] = summary.pop('Guadagno Netto Ora (€)')

            # Calcolo robusto del SoC finale
            last_op_row = df_result.iloc[-2]
            soc_initial = last_op_row.get('SoC Iniziale (%)', 0)
            soc_variation = last_op_row.get('Variazione SoC (%)', 0)
            summary['SoC Finale (%)'] = round(soc_initial + soc_variation, 2)

            all_run_results.append(summary)

    # =======================================================================
    # FASE 5: ANALISI E SALVATAGGIO DEI RISULTATI
    # =======================================================================
    if not all_run_results:
        print("\nNessun risultato da analizzare. Uscita.")
        return

    print("\nAnalisi dei risultati completata. Calcolo delle performance medie...")
    
    results_df = pd.DataFrame(all_run_results)
    
    metrics_to_analyze = [
        'Guadagno Netto (€)',
        'Costo Degradazione (€)',
        'Costo Ansia (€)',
        'SoC Finale (%)'
    ]
    
    # Raggruppa per strategia e calcola la media di ogni metrica
    average_performance = results_df.groupby('Strategia')[metrics_to_analyze].mean().round(4)
    
    print("\n" + "="*80)
    print("PERFORMANCE MEDIA DELLE STRATEGIE SU TUTTI GLI SCENARI")
    print("="*80)
    print(average_performance.to_string())
    print("="*80)

    # --- Salvataggio su file Excel ---
    output_dir = os.path.dirname(RL_PARAMS['q_table_file'])
    if not output_dir: output_dir = 'output'
    output_file = os.path.join(output_dir, "Analisi_Sensibilita_V2G.xlsx")
    
    try:
        with pd.ExcelWriter(output_file) as writer:
            results_df.to_excel(writer, sheet_name='Risultati Dettagliati', index=False)
            average_performance.to_excel(writer, sheet_name='Riepilogo Medie')
        
        print(f"\nRisultati dettagliati e medie salvati con successo in: '{output_file}'")
    except Exception as e:
        print(f"\nERRORE: Impossibile salvare il file Excel. Dettagli: {e}")

# =======================================================================
# ESECUZIONE DELLO SCRIPT
# =======================================================================
if __name__ == "__main__":
    run_sensitivity_analysis()
