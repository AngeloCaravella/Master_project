#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script per scaricare automaticamente i dati dal Mercato Elettrico
Compatibile con Python 3.7.9 e Chrome
"""

import os
import time
import logging
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException, 
    NoSuchElementException, 
    ElementClickInterceptedException,
    StaleElementReferenceException
)
from webdriver_manager.chrome import ChromeDriverManager

# Configurazione del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class MercatoElettricoDownloader:
    """Classe per scaricare i dati dal Mercato Elettrico"""
    
    def __init__(self, download_dir=None, max_retries=3, wait_time=30):
        """
        Inizializza il downloader
        
        Args:
            download_dir (str): Directory dove salvare i file scaricati
            max_retries (int): Numero massimo di tentativi in caso di errore
            wait_time (int): Tempo massimo di attesa in secondi per gli elementi
        """
        self.url = "https://www.mercatoelettrico.org/it-it/Home/Esiti/Elettricita/MGP/Esiti/PrezziZonali"
        self.max_retries = max_retries
        self.wait_time = wait_time
        
        # Imposta la directory di download
        if download_dir is None:
            self.download_dir = os.path.join(os.getcwd(), "downloads")
        else:
            self.download_dir = download_dir
            
        # Crea la directory se non esiste
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)
            
        logger.info(f"I file verranno scaricati in: {self.download_dir}")
        
        # Configura le opzioni di Chrome
        self.chrome_options = Options()
        self.chrome_options.add_experimental_option("prefs", {
            "download.default_directory": self.download_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True
        })
        
        # Opzioni aggiuntive per evitare problemi
        self.chrome_options.add_argument("--disable-extensions")
        self.chrome_options.add_argument("--disable-gpu")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        
        # Inizializza il driver
        self.driver = None
        
    def setup_driver(self):
        """Inizializza il driver di Chrome"""
        try:
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=self.chrome_options)
            self.driver.maximize_window()
            logger.info("Driver Chrome inizializzato con successo")
            return True
        except Exception as e:
            logger.error(f"Errore durante l'inizializzazione del driver: {str(e)}")
            return False
            
    def wait_for_element(self, by, value, timeout=None, condition=EC.presence_of_element_located):
        """
        Attende che un elemento sia presente nella pagina
        
        Args:
            by: Metodo di localizzazione (By.ID, By.XPATH, ecc.)
            value: Valore per la localizzazione
            timeout: Timeout in secondi (se None, usa self.wait_time)
            condition: Condizione di attesa
            
        Returns:
            L'elemento trovato o None in caso di errore
        """
        if timeout is None:
            timeout = self.wait_time
            
        try:
            element = WebDriverWait(self.driver, timeout).until(
                condition((by, value))
            )
            return element
        except TimeoutException:
            logger.warning(f"Timeout durante l'attesa dell'elemento: {value}")
            return None
        except Exception as e:
            logger.error(f"Errore durante l'attesa dell'elemento {value}: {str(e)}")
            return None
            
    def click_element_safely(self, element, retry_count=0):
        """
        Tenta di cliccare su un elemento in modo sicuro
        
        Args:
            element: Elemento da cliccare
            retry_count: Contatore dei tentativi
            
        Returns:
            True se il click è riuscito, False altrimenti
        """
        if retry_count >= 3:
            logger.error("Numero massimo di tentativi di click raggiunto")
            return False
            
        try:
            # Scorre fino all'elemento per assicurarsi che sia visibile
            self.driver.execute_script("arguments[0].scrollIntoView(true);", element)
            time.sleep(0.5)  # Breve pausa per permettere lo scroll
            
            # Tenta di cliccare con JavaScript se l'elemento è presente ma non cliccabile
            try:
                element.click()
            except (ElementClickInterceptedException, StaleElementReferenceException):
                logger.info("Tentativo di click con JavaScript")
                self.driver.execute_script("arguments[0].click();", element)
                
            time.sleep(1)  # Attesa breve dopo il click
            return True
        except Exception as e:
            logger.warning(f"Errore durante il click: {str(e)}. Tentativo {retry_count+1}/3")
            time.sleep(1)
            return self.click_element_safely(element, retry_count + 1)
            
    def accept_popup(self):
        """
        Accetta il popup iniziale delle condizioni d'uso
        
        Returns:
            True se l'accettazione è riuscita, False altrimenti
        """
        try:
            # Attendi che il popup sia visibile
            logger.info("In attesa del popup delle condizioni d'uso...")
            
            # Attendi che le checkbox siano presenti
            checkboxes = self.wait_for_element(
                By.XPATH, 
                "//input[@type='checkbox']", 
                timeout=15,
                condition=EC.presence_of_all_elements_located
            )
            
            if not checkboxes:
                logger.warning("Checkbox non trovate nel popup")
                return False
                
            # Seleziona tutte le checkbox
            for checkbox in checkboxes:
                if not checkbox.is_selected():
                    try:
                        checkbox.click()
                    except:
                        self.driver.execute_script("arguments[0].click();", checkbox)
                    logger.info("Checkbox selezionata")
                    time.sleep(0.5)
                    
            # Clicca sul pulsante per continuare
            continue_button = self.wait_for_element(
                By.XPATH, 
                "//button[contains(text(), 'CONTINUA SU MERCATOELETTRICO.ORG')]",
                condition=EC.element_to_be_clickable
            )
            
            if not continue_button:
                logger.warning("Pulsante 'CONTINUA' non trovato")
                return False
                
            success = self.click_element_safely(continue_button)
            if success:
                logger.info("Popup accettato con successo")
                time.sleep(2)  # Attesa per il caricamento della pagina
                return True
            else:
                logger.error("Impossibile cliccare sul pulsante 'CONTINUA'")
                return False
                
        except Exception as e:
            logger.error(f"Errore durante l'accettazione del popup: {str(e)}")
            return False
            
    def export_xlsx(self):
        """
        Clicca sul pulsante 'Esporta XLSX' e poi su 'Esporta tutte le zone XLSX'
        
        Returns:
            True se l'esportazione è riuscita, False altrimenti
        """
        try:
            # Attendi che il pulsante 'ESPORTA XLSX' sia presente e cliccabile
            logger.info("Cerco il pulsante 'ESPORTA XLSX'...")
            export_button = self.wait_for_element(
                By.XPATH, 
                "//button[contains(text(), 'ESPORTA XLSX')]",
                condition=EC.element_to_be_clickable
            )
            
            if not export_button:
                # Prova con un selettore alternativo
                export_button = self.wait_for_element(
                    By.CSS_SELECTOR, 
                    "button.btn.btn-success.btn-sm",
                    condition=EC.element_to_be_clickable
                )
                
            if not export_button:
                logger.error("Pulsante 'ESPORTA XLSX' non trovato")
                return False
                
            # Clicca sul pulsante 'ESPORTA XLSX'
            success = self.click_element_safely(export_button)
            if not success:
                logger.error("Impossibile cliccare sul pulsante 'ESPORTA XLSX'")
                return False
                
            logger.info("Pulsante 'ESPORTA XLSX' cliccato, attendo il menu a tendina...")
            time.sleep(2)  # Attesa per l'apertura del menu a tendina
            
            # Attendi che l'opzione 'Esporta tutte le zone XLSX' sia presente e cliccabile
            export_all_option = self.wait_for_element(
                By.XPATH, 
                "//a[contains(text(), 'Esporta tutte le zone XLSX')]",
                condition=EC.element_to_be_clickable
            )
            
            if not export_all_option:
                logger.error("Opzione 'Esporta tutte le zone XLSX' non trovata")
                return False
                
            # Clicca sull'opzione 'Esporta tutte le zone XLSX'
            success = self.click_element_safely(export_all_option)
            if not success:
                logger.error("Impossibile cliccare sull'opzione 'Esporta tutte le zone XLSX'")
                return False
                
            logger.info("Opzione 'Esporta tutte le zone XLSX' cliccata, download in corso...")
            
            # Attendi il completamento del download
            time.sleep(5)  # Attesa per l'inizio del download
            
            return True
            
        except Exception as e:
            logger.error(f"Errore durante l'esportazione XLSX: {str(e)}")
            return False
            
    def wait_for_download(self, timeout=30):
        """
        Attende il completamento del download
        
        Args:
            timeout: Timeout in secondi
            
        Returns:
            True se il download è completato, False altrimenti
        """
        start_time = time.time()
        
        # Controlla la presenza di file .crdownload o .tmp nella directory di download
        while time.time() - start_time < timeout:
            downloading_files = [f for f in os.listdir(self.download_dir) 
                               if f.endswith('.crdownload') or f.endswith('.tmp')]
            
            if not downloading_files:
                # Verifica che ci sia almeno un file XLSX nella directory
                xlsx_files = [f for f in os.listdir(self.download_dir) if f.endswith('.xlsx')]
                if xlsx_files:
                    logger.info(f"Download completato. File scaricati: {xlsx_files}")
                    return True
                    
            time.sleep(1)
            
        logger.warning(f"Timeout durante l'attesa del download dopo {timeout} secondi")
        return False
            
    def run(self):
        """
        Esegue il processo completo di download
        
        Returns:
            True se il processo è completato con successo, False altrimenti
        """
        retry_count = 0
        success = False
        
        while retry_count < self.max_retries and not success:
            if retry_count > 0:
                logger.info(f"Tentativo {retry_count+1}/{self.max_retries}")
                
            try:
                # Inizializza il driver
                if not self.setup_driver():
                    retry_count += 1
                    continue
                    
                # Apri la pagina
                logger.info(f"Apertura della pagina: {self.url}")
                self.driver.get(self.url)
                
                # Attendi il caricamento della pagina
                time.sleep(5)
                
                # Accetta il popup
                if not self.accept_popup():
                    logger.warning("Errore durante l'accettazione del popup, riprovo...")
                    retry_count += 1
                    self.driver.quit()
                    time.sleep(2)
                    continue
                    
                # Esporta i dati in XLSX
                if not self.export_xlsx():
                    logger.warning("Errore durante l'esportazione XLSX, riprovo...")
                    retry_count += 1
                    self.driver.quit()
                    time.sleep(2)
                    continue
                    
                # Attendi il completamento del download
                if not self.wait_for_download():
                    logger.warning("Errore durante l'attesa del download, riprovo...")
                    retry_count += 1
                    self.driver.quit()
                    time.sleep(2)
                    continue
                    
                # Se siamo arrivati qui, il processo è completato con successo
                success = True
                logger.info("Processo completato con successo!")
                
            except Exception as e:
                logger.error(f"Errore durante l'esecuzione: {str(e)}")
                retry_count += 1
                
            finally:
                # Chiudi il driver se è stato inizializzato
                if self.driver:
                    self.driver.quit()
                    
        return success
        
if __name__ == "__main__":
    # Percorso per il download dei file
    download_directory = os.path.join(os.getcwd(), "downloads")
    
    # Crea l'istanza del downloader e avvia il processo
    downloader = MercatoElettricoDownloader(
        download_dir=download_directory,
        max_retries=3,
        wait_time=30
    )
    
    result = downloader.run()
    
    if result:
        print("\n✅ Download completato con successo!")
        print(f"I file sono stati salvati in: {download_directory}")
    else:
        print("\n❌ Si è verificato un errore durante il download.")
        print("Controlla i log per maggiori dettagli.")
