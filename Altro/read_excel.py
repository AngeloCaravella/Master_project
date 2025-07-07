import pandas as pd

# Usiamo una stringa raw (r'...') per evitare problemi con i backslash di Windows
file_path = r'C:\Users\angel\OneDrive\Desktop\Progetto_tesi\Analisi_Sensibilita_Risultati.xlsx'

try:
    df_medie = pd.read_excel(file_path, sheet_name='Riepilogo Medie')
    df_dettagliati = pd.read_excel(file_path, sheet_name='Risultati Dettagliati')

    print('--- Riepilogo Medie ---')
    print(df_medie.to_string())
    print('\n--- Risultati Dettagliati (prime 20 righe) ---')
    print(df_dettagliati.head(20).to_string())

except FileNotFoundError:
    print(f"ERRORE: File non trovato a '{file_path}'")
except Exception as e:
    print(f"ERRORE: Impossibile leggere il file Excel. Dettagli: {e}")