# -*- coding: utf-8 -*-

# =======================================================================
# SCRIPT CONFIGURABILE PER ANALISI DI SENSIBILITÀ V2G
# Author: Angelo Caravella
# Version: 2.5
# Description: Questo script esegue un'analisi di sensibilità completa e
#              configurabile. Permette all'utente di scegliere la fonte
#              dei dati e il modello di degradazione della batteria tramite
#              argomenti da riga di comando.
#
# Esempio di Esecuzione:
# python run_configurable_analysis.py --file "path/al/tuo/file.xlsx" --zone "SUD" --degradation "lfp"
#
# =======================================================================

import numpy as np
import pandas as pd
import itertools
import os
import sys
import argparse
from tqdm import tqdm

# =======================================================================
# FASE 1: IMPORTAZIONE E CONTROLLO
# =======================================================================
try:
    from New import (V2GOptimizer, load_price_data, split_data, 
                     compare_strategies, save_results_to_excel,
                     RL_PARAMS, SIMULATION_PARAMS, VEHICLE_PARAMS)
except ImportError:
    print("ERRORE: Assicurati che il file 'New.py' sia nella stessa cartella.")
    sys.exit()

# =======================================================================
# FASE 2: DEFINIZIONE DEI PARAMETRI PER L'ANALISI DI SENSIBILITÀ
# =======================================================================
PARAM_RANGES = {
    'soc_min_utente': [0.2, 0.4, 0.6],
    'penalita_ansia': [0.005, 0.01, 0.02],
    'costo_batteria': [120 * 60, 150 * 60, 200 * 60],
    'soc_target_finale': [0.5, 0.7, 0.85]
}

# =======================================================================
# FASE 3: FUNZIONE PRINCIPALE PER L'ANALISI
# =======================================================================
def run_sensitivity_analysis(args):
    """
    Esegue l'analisi di sensibilità usando la configurazione fornita
    tramite argomenti da riga di comando.
    """
    print("=" * 80)
    print("AVVIO ANALISI DI SENSIBILITA' CONFIGURABILE SULLE STRATEGIE V2G")
    print("=" * 80)
    print("Configurazione per questa esecuzione:")
    print(f"  - File Prezzi: {args.file}")
    print(f"  - Zona di Test: {args.zone}")
    print(f"  - Modello Degradazione: {args.degradation.upper()}")
    print("-" * 80)

    # --- 1. Caricamento e Divisione Dati ---
    all_price_data = load_price_data(file_path=args.file)
    training_profiles, test_profile = split_data(all_price_data, test_zone=args.zone)

    # --- 2. Addestramento o Caricamento Agente RL ---
    # Modifica la configurazione del veicolo per il training in base alla scelta
    temp_vehicle_config = VEHICLE_PARAMS.copy()
    temp_vehicle_config['degradation_model'] = args.degradation
    
    temp_optimizer_for_training = V2GOptimizer(temp_vehicle_config, SIMULATION_PARAMS)
    q_table_file = RL_PARAMS['q_table_file'].replace('.npy', f'_{args.degradation}.npy')
    
    if os.path.exists(q_table_file):
        print(f"\nCaricamento Q-table esistente da '{q_table_file}'...")
        q_table = np.load(q_table_file)
    else:
        print(f"\nNessuna Q-table trovata. Avvio addestramento per modello '{args.degradation.upper()}'...")
        q_table = temp_optimizer_for_training.train_rl_agent(RL_PARAMS, training_profiles)
    
    # --- 3. Creazione delle combinazioni di parametri ---
    keys, values = zip(*PARAM_RANGES.items())
    parameter_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"\nVerranno eseguite {len(parameter_combinations)} simulazioni per ogni strategia.")
    
    all_run_results = []

    # --- 4. Esecuzione del ciclo di simulazioni ---
    for params in tqdm(parameter_combinations, desc=f"Analisi ({args.degradation.upper()})"):
        
        sim_config = SIMULATION_PARAMS.copy()
        sim_config.update({k: v for k, v in params.items() if k in sim_config})

        vehicle_config = VEHICLE_PARAMS.copy()
        vehicle_config.update({k: v for k, v in params.items() if k in vehicle_config})
        vehicle_config['degradation_model'] = args.degradation # Assicura che il modello scelto sia usato

        optimizer = V2GOptimizer(vehicle_config, sim_config)
        optimizer.set_prices_for_simulation(test_profile)

        strategies = {
            "Euristica Semplice": optimizer.run_heuristic_strategy,
            "Euristica LCVF": optimizer.run_lcvf_strategy,
            f"MPC (O={sim_config['mpc_horizon']}h)": lambda: optimizer.run_mpc_strategy(horizon=sim_config['mpc_horizon']),
            "Reinforcement Learning": lambda: optimizer.run_rl_strategy(q_table)
        }

        for name, func in strategies.items():
            original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            try:
                df_result = func()
            finally:
                sys.stdout.close()
                sys.stdout = original_stdout

            if df_result.empty: continue

            summary = df_result.iloc[-1].to_dict()
            summary.update(params)
            summary['Strategia'] = name
            
            if 'Guadagno Netto Ora (€)' in summary:
                summary['Guadagno Netto (€)'] = summary.pop('Guadagno Netto Ora (€)')

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
    
    average_performance = results_df.groupby('Strategia')[metrics_to_analyze].mean().round(4)
    
    print("\n" + "="*80)
    print(f"PERFORMANCE MEDIA DELLE STRATEGIE (Modello Degradazione: {args.degradation.upper()})")
    print("="*80)
    print(average_performance.to_string())
    print("="*80)

    output_dir = os.path.dirname(RL_PARAMS['q_table_file'])
    if not output_dir: output_dir = 'output'
    output_file = os.path.join(output_dir, f"Analisi_Sensibilita_{args.degradation.upper()}.xlsx")
    
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
    # --- Definizione degli argomenti da riga di comando ---
    parser = argparse.ArgumentParser(
        description="Esegue un'analisi di sensibilità configurabile per strategie V2G.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    script_dir = os.path.dirname(__file__) if '__file__' in locals() else os.getcwd()
    default_file_path = os.path.join(script_dir, "downloads", "PrezziZonali.xlsx")

    parser.add_argument(
        '--file', 
        type=str, 
        default=default_file_path,
        help="Percorso del file Excel contenente i dati dei prezzi."
    )
    parser.add_argument(
        '--zone', 
        type=str, 
        default="Italia",
        help="Nome della colonna/zona da usare come set di test."
    )
    parser.add_argument(
        '--degradation', 
        type=str, 
        default='nca',
        choices=['simple', 'lfp', 'nca'],
        help="Modello di degradazione della batteria da utilizzare per l'analisi."
    )
    
    # Esegui il parsing degli argomenti
    parsed_args = parser.parse_args()
    
    # Lancia la funzione principale con gli argomenti forniti
    run_sensitivity_analysis(parsed_args)
