# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import itertools
import os
import sys
from tqdm import tqdm

# =======================================================================
# FASE 1: IMPORTAZIONE E CONTROLLO
# =======================================================================
# Importa le classi e le funzioni necessarie dal tuo script originale.
# Assicurati che Mpc.py sia nella stessa cartella o in un percorso accessibile.
try:
    from Mpc import V2GOptimizer, load_price_data, RL_PARAMS, SIMULATION_PARAMS, VEHICLE_PARAMS
except ImportError:
    print("ERRORE: Assicurati che il file 'Mpc.py' sia nella stessa cartella.")
    sys.exit()

# =======================================================================
# FASE 2: DEFINIZIONE DEI PARAMETRI PER L'ANALISI
# =======================================================================
# Definiamo qui i range di valori da testare per l'analisi di sensibilità.
# Scegliamo valori sensati per evitare combinazioni irrealistiche.
PARAM_RANGES = {
    'soglia_ansia_soc': [0.2, 0.4, 0.6],       # Da utente "rischioso" a prudente
    'costo_degradazione': [0.01, 0.03, 0.05],  # Da basso ad alto costo di usura
    'penalita_ansia': [0.10, 0.20, 0.30],      # Da penalità leggera a severa
    'soc_target_finale': [0.5, 0.7, 0.85]      # Obiettivi di fine giornata
}

# =======================================================================
# FASE 3: FUNZIONE PRINCIPALE PER L'ANALISI
# =======================================================================
def run_sensitivity_analysis():
    """
    Esegue un'analisi di sensibilità completa, iterando su tutte le
    combinazioni dei parametri definiti in PARAM_RANGES.
    Questo script è stato riscritto per essere più robusto e risolvere
    l'errore 'KeyError' relativo a 'SoC Finale (%)'.
    """
    print("=" * 80)
    print("AVVIO ANALISI DI SENSIBILITA' SULLE STRATEGIE V2G")
    print("=" * 80)

    # --- Caricamento Dati (una sola volta) ---
    # Usiamo un percorso e una zona fissi per garantire la ripetibilità dell'analisi.
    script_dir = os.path.dirname(__file__) if '__file__' in locals() else os.getcwd()
    default_price_file = os.path.join(script_dir, "downloads", "PrezziZonali.xlsx")
    prezzi_per_zona = load_price_data(file_path=default_price_file, zone_name="Italia")
    if not prezzi_per_zona:
        sys.exit("Uscita. Dati dei prezzi non caricati.")

    # --- Addestramento o Caricamento Agente RL (una sola volta) ---
    # Creiamo un'istanza temporanea solo per gestire l'addestramento o il caricamento.
    temp_optimizer = V2GOptimizer(VEHICLE_PARAMS, SIMULATION_PARAMS, prezzi_per_zona)
    q_table_file = RL_PARAMS['q_table_file']
    
    if os.path.exists(q_table_file):
        print(f"\nCaricamento Q-table esistente da '{q_table_file}'...")
        q_table = np.load(q_table_file)
    else:
        print(f"\nNessuna Q-table trovata in '{q_table_file}'. Avvio addestramento una tantum...")
        q_table = temp_optimizer.train_rl_agent(RL_PARAMS)
    
    # --- Creazione di tutte le combinazioni di parametri ---
    keys, values = zip(*PARAM_RANGES.items())
    parameter_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"\nVerranno eseguite {len(parameter_combinations)} simulazioni per ogni strategia.")
    
    all_results = []

    # --- Esecuzione del ciclo di simulazioni ---
    # Usiamo tqdm per una barra di progresso che mostra l'avanzamento.
    for params in tqdm(parameter_combinations, desc="Simulazioni in corso"):
        
        # Crea una configurazione specifica per questa iterazione
        sim_config = SIMULATION_PARAMS.copy()
        sim_config['soc_min_utente'] = params['soglia_ansia_soc']
        sim_config['penalita_ansia'] = params['penalita_ansia']
        sim_config['soc_target_finale'] = params['soc_target_finale']

        vehicle_config = VEHICLE_PARAMS.copy()
        vehicle_config['costo_degradazione'] = params['costo_degradazione']

        # Inizializza l'ottimizzatore con i parametri correnti
        optimizer = V2GOptimizer(vehicle_config, sim_config, prezzi_per_zona)

        # Definisci le strategie da eseguire
        strategies = {
            "Euristica Semplice": optimizer.run_heuristic_strategy,
            "Euristica LCVF": optimizer.run_lcvf_strategy,
            f"MPC (O={SIMULATION_PARAMS['mpc_horizon']}h)": lambda: optimizer.run_mpc_strategy(horizon=SIMULATION_PARAMS['mpc_horizon']),
            "MPC (O=24h)": lambda: optimizer.run_mpc_strategy(horizon=24),
            "Reinforcement Learning": lambda: optimizer.run_rl_strategy(q_table)
        }

        for name, func in strategies.items():
            # Sopprimiamo la stampa dettagliata per pulire l'output della console
            original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            try:
                # Esegui la simulazione e ottieni il DataFrame dei risultati
                df_result = func()
            finally:
                # Ripristina sempre l'output standard, anche in caso di errore
                sys.stdout.close()
                sys.stdout = original_stdout

            # --- ESTRAZIONE E PULIZIA DEI DATI (PARTE CRUCIALE DELLA CORREZIONE) ---
            
            # 1. Estrai la riga di riepilogo ("TOTALE"), che è sempre l'ultima
            summary = df_result.iloc[-1].to_dict()

            # 2. Aggiungi i parametri di questa simulazione e il nome della strategia
            summary.update(params)
            summary['Strategia'] = name
            
            # 3. Standardizza il nome della colonna del guadagno
            if 'Guadagno Netto Ora (€)' in summary:
                summary['Guadagno Netto (€)'] = summary.pop('Guadagno Netto Ora (€)')
            elif 'Ricompensa (€)' in summary:
                summary['Guadagno Netto (€)'] = summary.pop('Ricompensa (€)')

            # 4. CALCOLO ROBUSTO DEL SOC FINALE
            # Il SoC finale non è nella riga "TOTALE". Lo calcoliamo dall'ultima ora di operazione,
            # che è la penultima riga del DataFrame (df_result.iloc[-2]).
            # Questo approccio è sicuro e previene il KeyError.
            last_operational_row = df_result.iloc[-2]
            soc_initial_last_hour = last_operational_row.get('SoC Iniziale (%)', 0)
            soc_variation_last_hour = last_operational_row.get('Variazione SoC (%)', 0)
            
            # Il SoC finale è il SoC iniziale dell'ultima ora + la sua variazione
            final_soc = soc_initial_last_hour + soc_variation_last_hour
            summary['SoC Finale (%)'] = round(final_soc, 2)

            # 5. Aggiungi il dizionario pulito e completo alla lista dei risultati
            all_results.append(summary)

    # =======================================================================
    # FASE 4: ANALISI E SALVATAGGIO DEI RISULTATI
    # =======================================================================
    print("\nAnalisi dei risultati completata. Calcolo delle performance medie...")
    
    # Crea il DataFrame finale con tutti i risultati
    results_df = pd.DataFrame(all_results)
    
    # Seleziona le colonne (metriche) che vogliamo analizzare
    metrics_to_analyze = [
        'Guadagno Netto (€)',
        'Costo Degradazione (€)',
        'Costo Ansia (€)',
        'SoC Finale (%)'  # Ora questa colonna esisterà sempre
    ]
    
    # Raggruppa per strategia e calcola la media di ogni metrica
    average_performance = results_df.groupby('Strategia')[metrics_to_analyze].mean().round(4)
    
    print("\n" + "="*80)
    print("PERFORMANCE MEDIA DELLE STRATEGIE SU TUTTI GLI SCENARI")
    print("="*80)
    print(average_performance.to_string())
    print("="*80)

    # --- Salvataggio su file Excel ---
    output_file = "Analisi_Sensibilita_Risultati_Corretto.xlsx"
    try:
        with pd.ExcelWriter(output_file) as writer:
            # Salva i dati dettagliati di ogni singola simulazione
            results_df.to_excel(writer, sheet_name='Risultati Dettagliati', index=False)
            # Salva la tabella con le medie delle performance
            average_performance.to_excel(writer, sheet_name='Riepilogo Medie')
        
        print(f"\nRisultati dettagliati e medie salvati con successo in: '{output_file}'")
    except Exception as e:
        print(f"\nERRORE: Impossibile salvare il file Excel. Dettagli: {e}")


# =======================================================================
# ESECUZIONE DELLO SCRIPT
# =======================================================================
if __name__ == "__main__":
    run_sensitivity_analysis()
