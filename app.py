
import os
import uuid
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

# ============================================================
# Setup iniziale
# ============================================================
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY non trovata. Crea un file .env con la tua chiave OpenAI.")
    st.stop()

st.set_page_config(page_title="EcoBot – Soluzioni EcoGrid", page_icon="🧐")
st.title("🧐EcoBot – l'analista energetico🔆")
st.caption("Formula una domanda sui tuoi dati di produzione solare e consumo domestico.")

CARTELLA_GRAFICI = "grafici"
os.makedirs(CARTELLA_GRAFICI, exist_ok=True)

# Fattore di emissione medio della rete elettrica italiana: 0,216 kg CO2/kWh
# (fonte: ISPRA, dato 2023 sulla produzione elettrica nazionale lorda - valore indicativo)
FATTORE_EMISSIONE_CO2_KWH = 0.216

# Località di fallback se la geocodifica della città inserita dall'utente fallisce
LOCALITA_DEFAULT = {"nome": "Roma", "lat": 41.9028, "lon": 12.4964}

# File di persistenza della gamification (sopravvivono ai riavvii di Streamlit/kernel)
FILE_MISSIONE = "missione_attiva.json"
FILE_BADGE = "badge_sbloccati.json"

# Costanti delle missioni giornaliere
PERCENTUALE_TARGET_SOLARE = 20
PERCENTUALE_TARGET_RISPARMIO = 15
PERCENTUALE_TARGET_RETE = 15
BADGE_MISSIONE_SOLARE = "Domatore di Sole"
BADGE_MISSIONE_RISPARMIO = "Guardiano del Risparmio"
BADGE_MISSIONE_RETE = "Alleggerisci la Rete"
SOGLIA_PICCO_SERALE = 1.3  # la fascia serale deve avere un consumo medio orario almeno 1.3 volte
                            # quello medio complessivo per essere considerata un "picco pronunciato"


# ============================================================
# Data store
# ============================================================
@dataclass
class DatiEcoBot:
    """Contenitore dei dati puliti su cui operano i tool dell'agente."""
    produzione: pd.DataFrame = None
    consumo: pd.DataFrame = None
    log_pulizia: list = field(default_factory=list)
    grafici_generati: list = field(default_factory=list)


# ============================================================
# Motore di pulizia e helper individuazione periodo
# ============================================================
def _pulisci_serie(df, colonna_valore, nome_dataset):
    messaggi = []
    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    duplicati = df["timestamp"].duplicated().sum()
    if duplicati > 0:
        df = df.drop_duplicates(subset="timestamp", keep="first")
        messaggi.append(f"[{nome_dataset}] Rimossi {duplicati} timestamp duplicati.")
    df = df.sort_values("timestamp").reset_index(drop=True)

    n_nan = df[colonna_valore].isna().sum()
    if n_nan > 0:
        df[colonna_valore] = df[colonna_valore].interpolate(method="linear", limit_direction="both")
        messaggi.append(f"[{nome_dataset}] Interpolati {n_nan} valori mancanti (probabile guasto sensore).")

    n_negativi = (df[colonna_valore] < 0).sum()
    if n_negativi > 0:
        df.loc[df[colonna_valore] < 0, colonna_valore] = 0.0
        messaggi.append(f"[{nome_dataset}] Corretti {n_negativi} valori negativi impossibili (portati a 0).")

    if len(df) > 1:
        attesi = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h")
        mancanti = attesi.difference(df["timestamp"])
        if len(mancanti) > 0:
            messaggi.append(f"[{nome_dataset}] Rilevati {len(mancanti)} intervalli orari mancanti nella serie.")

    if not messaggi:
        messaggi.append(f"[{nome_dataset}] Nessuna anomalia rilevata.")

    return df, messaggi


def _filtra_periodo(df, data_inizio=None, data_fine=None):
    if data_inizio:
        df = df[df["timestamp"] >= pd.to_datetime(data_inizio)]
    if data_fine:
        df = df[df["timestamp"] < pd.to_datetime(data_fine) + pd.Timedelta(days=1)]
    return df


def _calcola_kpi(prod: pd.DataFrame, cons: pd.DataFrame):
    """Calcola i valori grezzi dei KPI energetici (autoconsumo, autosufficienza, surplus) a partire
    da due dataframe già filtrati per periodo. Usata sia dal tool dell'agente sia dal pannello KPI
    in Streamlit, per non duplicare la logica di calcolo in due posti diversi."""
    merged = pd.merge(prod, cons, on="timestamp", how="inner")
    if merged.empty:
        return None
    totale_prod = merged["kwh_produced"].sum()
    totale_cons = merged["kwh_consumed"].sum()
    autoconsumo_orario = merged[["kwh_produced", "kwh_consumed"]].min(axis=1)
    energia_autoconsumata = autoconsumo_orario.sum()
    return {
        "totale_prod": totale_prod,
        "totale_cons": totale_cons,
        "energia_autoconsumata": energia_autoconsumata,
        "indice_autoconsumo": (energia_autoconsumata / totale_prod * 100) if totale_prod > 0 else 0.0,
        "indice_autosufficienza": (energia_autoconsumata / totale_cons * 100) if totale_cons > 0 else 0.0,
        "surplus_ceduto": totale_prod - energia_autoconsumata,
        "percentuale_surplus": ((totale_prod - energia_autoconsumata) / totale_prod * 100) if totale_prod > 0 else 0.0,
    }


def _indice_autoconsumo_periodo(dati: DatiEcoBot, data_inizio, data_fine):
    """Restituisce l'indice di autoconsumo (%) nel periodo indicato (date pandas/Timestamp),
    o None se non ci sono abbastanza dati nel periodo. Riusa _filtra_periodo e la stessa logica
    di _calcola_kpi, isolata sul solo indice di autoconsumo."""
    prod = _filtra_periodo(dati.produzione, data_inizio.strftime("%Y-%m-%d"), data_fine.strftime("%Y-%m-%d"))
    cons = _filtra_periodo(dati.consumo, data_inizio.strftime("%Y-%m-%d"), data_fine.strftime("%Y-%m-%d"))
    kpi = _calcola_kpi(prod, cons)
    if kpi is None or kpi["totale_prod"] <= 0:
        return None
    return kpi["indice_autoconsumo"]


def _messaggio_eco_score(dati: DatiEcoBot):
    """Costruisce il messaggio di eco score settimanale (confronto dell'indice di autoconsumo tra
    gli ultimi 7 giorni disponibili e i 7 precedenti). Riusato sia dal tool conversazionale
    punteggio_eco_settimanale sia dal banner sempre visibile in Streamlit, per non duplicare la
    logica in due posti diversi. Ritorna None se mancano dati sufficienti (almeno 14 giorni)."""
    if dati.produzione is None or dati.consumo is None:
        return None

    ultimo_timestamp = max(dati.produzione["timestamp"].max(), dati.consumo["timestamp"].max())
    fine_corrente = ultimo_timestamp.normalize()
    inizio_corrente = fine_corrente - pd.Timedelta(days=6)
    fine_precedente = inizio_corrente - pd.Timedelta(days=1)
    inizio_precedente = fine_precedente - pd.Timedelta(days=6)

    autoconsumo_corrente = _indice_autoconsumo_periodo(dati, inizio_corrente, fine_corrente)
    autoconsumo_precedente = _indice_autoconsumo_periodo(dati, inizio_precedente, fine_precedente)

    if autoconsumo_corrente is None or autoconsumo_precedente is None:
        return None

    differenza = autoconsumo_corrente - autoconsumo_precedente

    if differenza > 0.5:
        return (
            f"Ottimo lavoro! Questa settimana sei salito al {autoconsumo_corrente:.0f}% di "
            f"autoconsumo, rispetto al {autoconsumo_precedente:.0f}% della settimana precedente."
        )
    elif differenza < -0.5:
        return (
            f"Questa settimana l'autoconsumo è sceso al {autoconsumo_corrente:.0f}% "
            f"(era {autoconsumo_precedente:.0f}% la settimana scorsa). Prova a spostare qualche "
            f"consumo nelle ore di maggior produzione solare per recuperare terreno."
        )
    else:
        return (
            f"Questa settimana l'autoconsumo è stabile al {autoconsumo_corrente:.0f}%, "
            f"in linea con la settimana precedente."
        )


def geocodifica_citta(nome_citta: str):
    """Converte il nome di una città in coordinate geografiche (lat/lon) usando l'API di geocoding
    gratuita di Open-Meteo. Ritorna un dizionario {'nome', 'lat', 'lon'}, oppure None se la città
    non viene trovata o l'API non risponde."""
    try:
        risposta = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": nome_citta, "count": 1, "language": "it"},
            timeout=10,
        )
        risposta.raise_for_status()
        risultati = risposta.json().get("results")
        if not risultati:
            return None
        primo = risultati[0]
        return {"nome": primo["name"], "lat": primo["latitude"], "lon": primo["longitude"]}
    except Exception:
        return None


# ============================================================
# Helper di persistenza per la gamification (missioni + badge)
# ============================================================
def _carica_missione():
    """Legge la missione attiva salvata su disco (se esiste), per farla sopravvivere ai riavvii
    di Streamlit. Ritorna None se non c'è nessuna missione attiva o il file non esiste/è corrotto."""
    if not os.path.exists(FILE_MISSIONE):
        return None
    try:
        with open(FILE_MISSIONE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _salva_missione(missione):
    """Salva su disco la missione attiva (dict), oppure la cancella se missione è None."""
    if missione is None:
        if os.path.exists(FILE_MISSIONE):
            os.remove(FILE_MISSIONE)
        return
    with open(FILE_MISSIONE, "w") as f:
        json.dump(missione, f, indent=2)


def _finestra_punta_solare(produzione):
    """Individua la fascia oraria di punta della produzione solare (picco ±1 ora) a partire dai dati
    storici di produzione dell'utente, invece di usare un orario fisso uguale per tutti gli impianti.
    Ritorna una tupla (ora_inizio, ora_fine), entrambe incluse."""
    medie_orarie = produzione.groupby(produzione["timestamp"].dt.hour)["kwh_produced"].mean()
    ora_picco = int(medie_orarie.idxmax())
    return max(0, ora_picco - 1), min(23, ora_picco + 1)


def _finestra_punta_consumo_serale(consumo):
    """Individua la fascia oraria di punta del consumo serale (picco ±1 ora, cercato solo tra le
    17:00 e le 23:00 per restare nell'orario tipico di punta della rete) a partire dai dati storici
    di consumo dell'utente. Ritorna una tupla (ora_inizio, ora_fine), entrambe incluse."""
    consumo_serale = consumo[consumo["timestamp"].dt.hour >= 17]
    medie_orarie = consumo_serale.groupby(consumo_serale["timestamp"].dt.hour)["kwh_consumed"].mean()
    ora_picco = int(medie_orarie.idxmax())
    return max(17, ora_picco - 1), min(23, ora_picco + 1)


def _consumo_medio_finestra(consumo, ora_inizio, ora_fine):
    """Consumo medio giornaliero (kWh) nella fascia oraria indicata (estremi inclusi), calcolato sulla
    media di tutti i giorni disponibili nei dati storici. Usato come 'consumo abituale' di riferimento
    per valutare le missioni giornaliere."""
    in_fascia = consumo[(consumo["timestamp"].dt.hour >= ora_inizio) & (consumo["timestamp"].dt.hour <= ora_fine)]
    per_giorno = in_fascia.groupby(in_fascia["timestamp"].dt.date)["kwh_consumed"].sum()
    return float(per_giorno.mean()) if not per_giorno.empty else 0.0


def _descrivi_missione_attiva(missione):
    """Costruisce una descrizione testuale della missione attiva (dict caricato da _carica_missione),
    adattandosi al tipo di missione (fascia oraria specifica o intera giornata) e alla direzione del
    target (aumento o riduzione). Riusata sia da genera_missione_giornaliera sia da verifica_missione."""
    verbo = "sposta almeno" if missione["direzione"] == "aumento" else "riduci di almeno"
    if missione["ora_inizio"] is not None:
        dove = f"nella fascia {missione['ora_inizio']}:00-{missione['ora_fine']}:00"
    else:
        dove = "sul totale della giornata"
    return (
        f"C'è già una missione attiva per il {missione['data_target']}: {verbo} il "
        f"{missione['percentuale_target']}% del consumo {dove} per sbloccare il badge "
        f"'{missione['badge']}'. Torna a chiedermi il risultato quando avrai caricato i dati aggiornati "
        "che includono quella data."
    )


def _consumo_giorno(consumo, data, ora_inizio=None, ora_fine=None):
    """Consumo totale (kWh) in una specifica data, opzionalmente ristretto a una fascia oraria
    (estremi inclusi). Ritorna None se quella data non è presente nei dati caricati."""
    del_giorno = consumo[consumo["timestamp"].dt.date == data]
    if ora_inizio is not None:
        del_giorno = del_giorno[(del_giorno["timestamp"].dt.hour >= ora_inizio) & (del_giorno["timestamp"].dt.hour <= ora_fine)]
    if del_giorno.empty:
        return None
    return float(del_giorno["kwh_consumed"].sum())


def _carica_badge():
    """Legge l'elenco dei badge sbloccati finora (persistito su disco). Ritorna una lista vuota se
    il file non esiste o è corrotto."""
    if not os.path.exists(FILE_BADGE):
        return []
    try:
        with open(FILE_BADGE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _salva_badge(badge_sbloccati):
    """Salva su disco l'elenco aggiornato dei badge sbloccati."""
    with open(FILE_BADGE, "w") as f:
        json.dump(badge_sbloccati, f, indent=2)


# ============================================================
# Factory dei tool (pulisci_dati in questo caso legge da file
# caricati in Streamlit invece che da percorsi fissi su disco)
# ============================================================
def crea_tool_pulizia(dati, file_produzione, file_consumo):
    @tool
    def pulisci_dati() -> str:
        """Carica i CSV di produzione solare e consumo domestico, individua e corregge automaticamente
        le anomalie tipiche dei sensori IoT (valori mancanti, valori negativi impossibili, timestamp
        duplicati o mancanti), e aggiorna il data store con i dataset puliti. Va chiamato come primo
        passo prima di qualsiasi analisi se i dati non sono ancora stati caricati."""
        file_produzione.seek(0)
        file_consumo.seek(0)
        df_prod_raw = pd.read_csv(file_produzione)
        df_cons_raw = pd.read_csv(file_consumo)

        dati.produzione, msg_prod = _pulisci_serie(df_prod_raw, "kwh_produced", "produzione")
        dati.consumo, msg_cons = _pulisci_serie(df_cons_raw, "kwh_consumed", "consumo")
        dati.log_pulizia = msg_prod + msg_cons

        return "Dati caricati e puliti con successo.\n" + "\n".join(dati.log_pulizia)

    return pulisci_dati


def crea_tool_produzione(dati):
    @tool
    def calcola_produzione(data_inizio: str = "", data_fine: str = "") -> str:
        """Calcola l'energia solare totale prodotta dall'azienda in un intervallo di date.
        Args:
            data_inizio: data di inizio in formato YYYY-MM-DD (vuota = dall'inizio dei dati disponibili).
            data_fine: data di fine in formato YYYY-MM-DD, inclusa (vuota = fino alla fine dei dati disponibili).
        """
        if dati.produzione is None:
            return "Nessun dato disponibile: chiama prima il tool pulisci_dati."
        df = _filtra_periodo(dati.produzione, data_inizio or None, data_fine or None)
        if df.empty:
            return "Nessun dato di produzione nel periodo richiesto."
        totale = df["kwh_produced"].sum()
        media_oraria = df["kwh_produced"].mean()
        return (
            f"Produzione totale: {totale:.2f} kWh su {len(df)} ore "
            f"({df['timestamp'].min().date()} - {df['timestamp'].max().date()}), "
            f"media oraria {media_oraria:.2f} kWh."
        )
    return calcola_produzione


def crea_tool_consumo(dati):
    @tool
    def calcola_consumo(data_inizio: str = "", data_fine: str = "") -> str:
        """Calcola l'energia totale consumata dalla famiglia in un intervallo di date.
        Args:
            data_inizio: data di inizio in formato YYYY-MM-DD (vuota = dall'inizio dei dati disponibili).
            data_fine: data di fine in formato YYYY-MM-DD, inclusa (vuota = fino alla fine dei dati disponibili).
        """
        if dati.consumo is None:
            return "Nessun dato disponibile: chiama prima il tool pulisci_dati."
        df = _filtra_periodo(dati.consumo, data_inizio or None, data_fine or None)
        if df.empty:
            return "Nessun dato di consumo nel periodo richiesto."
        totale = df["kwh_consumed"].sum()
        media_oraria = df["kwh_consumed"].mean()
        return (
            f"Consumo totale: {totale:.2f} kWh su {len(df)} ore "
            f"({df['timestamp'].min().date()} - {df['timestamp'].max().date()}), "
            f"media oraria {media_oraria:.2f} kWh."
        )
    return calcola_consumo


def crea_tool_confronto(dati):
    @tool
    def confronta_produzione_consumo(data_inizio: str = "", data_fine: str = "") -> str:
        """Confronta l'energia prodotta con quella consumata in un intervallo di date: dice se c'è
        surplus (energia in eccesso) o deficit, e indica il giorno con il maggior surplus.
        Args:
            data_inizio: data di inizio in formato YYYY-MM-DD (opzionale).
            data_fine: data di fine in formato YYYY-MM-DD, inclusa (opzionale).
        """
        if dati.produzione is None or dati.consumo is None:
            return "Nessun dato disponibile: chiama prima il tool pulisci_dati."

        prod = _filtra_periodo(dati.produzione, data_inizio or None, data_fine or None)
        cons = _filtra_periodo(dati.consumo, data_inizio or None, data_fine or None)
        merged = pd.merge(prod, cons, on="timestamp", how="inner")
        if merged.empty:
            return "Nessun dato disponibile nel periodo richiesto."

        totale_prod = merged["kwh_produced"].sum()
        totale_cons = merged["kwh_consumed"].sum()
        bilancio = totale_prod - totale_cons

        merged["giorno"] = merged["timestamp"].dt.date
        giornaliero = merged.groupby("giorno").agg(
            prodotto=("kwh_produced", "sum"), consumato=("kwh_consumed", "sum")
        )
        giornaliero["surplus"] = giornaliero["prodotto"] - giornaliero["consumato"]
        giorno_top = giornaliero["surplus"].idxmax()

        esito = "surplus (energia in eccesso)" if bilancio >= 0 else "deficit (consumato più di quanto prodotto)"
        return (
            f"Nel periodo: prodotto {totale_prod:.2f} kWh, consumato {totale_cons:.2f} kWh -> {esito} "
            f"di {abs(bilancio):.2f} kWh. Giorno con il maggior surplus: {giorno_top} "
            f"({giornaliero.loc[giorno_top, 'surplus']:.2f} kWh in eccesso)."
        )
    return confronta_produzione_consumo


def crea_tool_andamento_orario(dati):
    @tool
    def andamento_orario(serie: str, data_inizio: str = "", data_fine: str = "") -> str:
        """Calcola il valore medio per ogni ora del giorno (0-23) di produzione o consumo, utile per
        capire in quali fasce orarie si consuma o produce di più.
        Args:
            serie: 'produzione' oppure 'consumo'.
            data_inizio: data di inizio in formato YYYY-MM-DD (opzionale).
            data_fine: data di fine in formato YYYY-MM-DD, inclusa (opzionale).
        """
        if dati.produzione is None or dati.consumo is None:
            return "Nessun dato disponibile: chiama prima il tool pulisci_dati."

        if serie == "produzione":
            df, colonna = dati.produzione, "kwh_produced"
        elif serie == "consumo":
            df, colonna = dati.consumo, "kwh_consumed"
        else:
            return "Valore 'serie' non valido: usa 'produzione' o 'consumo'."

        df = _filtra_periodo(df, data_inizio or None, data_fine or None)
        if df.empty:
            return "Nessun dato nel periodo richiesto."

        medie_orarie = df.groupby(df["timestamp"].dt.hour)[colonna].mean().round(2)
        ora_picco = medie_orarie.idxmax()
        righe = [f"{ora}:00 -> {valore:.2f} kWh medi" for ora, valore in medie_orarie.items()]
        return (
            f"Andamento orario medio di {serie}:\n" + "\n".join(righe) +
            f"\n\nOra con il valore più alto: {ora_picco}:00 ({medie_orarie[ora_picco]:.2f} kWh medi)."
        )
    return andamento_orario


def crea_tool_anomalie(dati):
    @tool
    def individua_anomalie() -> str:
        """Riporta le anomalie rilevate durante la pulizia dei dati (es. guasti ai sensori, valori
        mancanti, valori impossibili corretti). Usalo per rispondere a domande come 'c'è stato un
        guasto ai sensori?'."""
        if not dati.log_pulizia:
            return "Nessuna pulizia dati ancora eseguita: chiama prima il tool pulisci_dati."
        return "Anomalie rilevate durante la pulizia dei dati:\n" + "\n".join(dati.log_pulizia)
    return individua_anomalie


def crea_tool_kpi_energetici(dati):
    @tool
    def calcola_kpi_energetici(data_inizio: str = "", data_fine: str = "") -> str:
        """Calcola i principali indicatori di performance (KPI) energetici in un intervallo di date:
        indice di autoconsumo, indice di autosufficienza e percentuale di surplus ceduto alla rete.
        Usalo per rispondere a domande su quanto viene "autoconsumato" o condiviso con la comunità
        energetica, non solo su totali semplici di produzione/consumo.
        Args:
            data_inizio: data di inizio in formato YYYY-MM-DD (opzionale).
            data_fine: data di fine in formato YYYY-MM-DD, inclusa (opzionale).
        """
        if dati.produzione is None or dati.consumo is None:
            return "Nessun dato disponibile: chiama prima il tool pulisci_dati."
        prod = _filtra_periodo(dati.produzione, data_inizio or None, data_fine or None)
        cons = _filtra_periodo(dati.consumo, data_inizio or None, data_fine or None)
        kpi = _calcola_kpi(prod, cons)
        if kpi is None:
            return "Nessun dato disponibile nel periodo richiesto."

        return (
            f"KPI energetici nel periodo:\n"
            f"- Indice di autoconsumo: {kpi['indice_autoconsumo']:.1f}% "
            f"(quota della produzione usata direttamente sul posto: {kpi['energia_autoconsumata']:.2f} kWh su {kpi['totale_prod']:.2f} kWh prodotti).\n"
            f"- Indice di autosufficienza: {kpi['indice_autosufficienza']:.1f}% "
            f"(quota del consumo coperta dalla produzione propria, su {kpi['totale_cons']:.2f} kWh consumati).\n"
            f"- Surplus ceduto alla rete/comunità: {kpi['surplus_ceduto']:.2f} kWh ({kpi['percentuale_surplus']:.1f}% della produzione totale)."
        )
    return calcola_kpi_energetici


def crea_tool_co2(dati):
    @tool
    def co2_risparmiata(data_inizio: str = "", data_fine: str = "") -> str:
        """Stima la CO2 evitata grazie all'energia rinnovabile prodotta in un intervallo di date,
        rispetto a se quella stessa energia fosse stata prelevata dalla rete elettrica tradizionale
        (mix medio italiano). Considera tutta l'energia prodotta, non solo quella autoconsumata, perché
        anche il surplus condiviso con la comunità energetica evita emissioni altrove nella rete.
        Args:
            data_inizio: data di inizio in formato YYYY-MM-DD (opzionale).
            data_fine: data di fine in formato YYYY-MM-DD, inclusa (opzionale).
        """
        if dati.produzione is None:
            return "Nessun dato disponibile: chiama prima il tool pulisci_dati."
        prod = _filtra_periodo(dati.produzione, data_inizio or None, data_fine or None)
        if prod.empty:
            return "Nessun dato di produzione nel periodo richiesto."

        totale_prod = prod["kwh_produced"].sum()
        co2_evitata_kg = totale_prod * FATTORE_EMISSIONE_CO2_KWH

        return (
            f"Nel periodo, la produzione di {totale_prod:.2f} kWh di energia rinnovabile ha evitato "
            f"circa {co2_evitata_kg:.1f} kg di CO2 rispetto all'uso della rete elettrica tradizionale "
            f"(stima basata su un fattore medio di {FATTORE_EMISSIONE_CO2_KWH} kg CO2/kWh, fonte ISPRA)."
        )
    return co2_risparmiata


def crea_tool_eco_score(dati):
    @tool
    def punteggio_eco_settimanale() -> str:
        """
        Confronta l'indice di autoconsumo degli ultimi 7 giorni disponibili nei dati con
        quello dei 7 giorni precedenti, per mostrare il progresso settimana su settimana
        (una sorta di "eco score"). Usalo per domande tipo "come sto andando questa
        settimana?", "sono migliorato rispetto alla settimana scorsa?" o "dammi il mio
        punteggio eco".
        """
        if dati.produzione is None or dati.consumo is None:
            return "Nessun dato disponibile: chiama prima il tool pulisci_dati."
        messaggio = _messaggio_eco_score(dati)
        if messaggio is None:
            return "Non ci sono abbastanza dati storici (servono almeno 14 giorni) per calcolare un confronto settimana su settimana."
        return messaggio
    return punteggio_eco_settimanale


def crea_tool_meteo(localita: dict):
    """Factory: crea il tool di previsione meteo legato a una specifica località (lat/lon).
    Passare la località dall'esterno (invece di una costante globale) evita che, con più utenti
    connessi contemporaneamente, la città scelta da uno influenzi le previsioni mostrate a un altro."""
    @tool
    def previsione_meteo(giorni: int = 3) -> str:
        """Restituisce le previsioni meteo (temperatura, radiazione solare attesa, probabilità di pioggia)
        per i prossimi giorni nella località dell'utente, con etichetta "oggi/domani/dopodomani/tra N
        giorni" per ogni giornata. Usalo per consigli proattivi su quando conviene usare elettrodomestici,
        o per stimare quanto varierà la produzione solare nei giorni a venire (i CSV contengono solo dati
        storici passati, quindi per parlare del futuro serve questo tool).
        Args:
            giorni: quante giornate restituire A PARTIRE DA OGGI incluso (il primo giorno restituito è
                sempre oggi), da 1 a 7. Es. per rispondere su "domani" chiedi almeno giorni=2, così la
                risposta includerà anche la voce etichettata "domani".
        """
        giorni = max(1, min(giorni, 7))
        try:
            risposta = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": localita["lat"],
                    "longitude": localita["lon"],
                    "daily": "temperature_2m_max,temperature_2m_min,shortwave_radiation_sum,precipitation_probability_max",
                    "timezone": "Europe/Rome",
                    "forecast_days": giorni,
                },
                timeout=10,
            )
            risposta.raise_for_status()
            dati_meteo = risposta.json()["daily"]
        except Exception as errore:
            return (
                f"Impossibile recuperare le previsioni meteo al momento ({errore}). "
                "Rispondi comunque basandoti sui dati storici disponibili, senza inventare un meteo futuro."
            )

        oggi = date.today()
        etichette_relative = {0: "oggi", 1: "domani", 2: "dopodomani"}

        righe = []
        for i, giorno in enumerate(dati_meteo["time"]):
            scarto = (date.fromisoformat(giorno) - oggi).days
            etichetta = etichette_relative.get(scarto, f"tra {scarto} giorni")
            righe.append(
                f"{etichetta} ({giorno}): {dati_meteo['temperature_2m_min'][i]:.0f}-{dati_meteo['temperature_2m_max'][i]:.0f}°C, "
                f"radiazione solare attesa {dati_meteo['shortwave_radiation_sum'][i]:.1f} MJ/m², "
                f"probabilità pioggia {dati_meteo['precipitation_probability_max'][i]:.0f}%"
            )

        return (
            f"Previsioni meteo per {localita['nome']}:\n" + "\n".join(righe) +
            "\n\nRadiazione solare più alta = giornata più soleggiata, aspettati più produzione fotovoltaica; "
            "radiazione bassa (giornata nuvolosa/piovosa) = produzione ridotta."
        )

    return previsione_meteo


def _richiedi_produzione_forecast(config_impianto):
    """Interroga Forecast.Solar e ritorna una tupla (dati, errore):
    - in caso di successo: (dict {data 'YYYY-MM-DD': wh prodotti quel giorno}, None)
    - in caso di errore di rete/HTTP: (None, messaggio di errore descrittivo)
    - in caso di risposta senza dati validi: (None, None)
    Condivisa dal tool conversazionale previsione_produzione e dal generatore di missioni giornaliere,
    per non duplicare la chiamata all'API esterna in due punti diversi."""
    lat = config_impianto["lat"]
    lon = config_impianto["lon"]
    dec = config_impianto.get("dec", 30)
    az = config_impianto.get("az", 0)
    kwp = config_impianto["kwp"]

    url = f"https://api.forecast.solar/estimate/{lat}/{lon}/{dec}/{az}/{kwp}"

    try:
        risposta = requests.get(url, timeout=10)
        risposta.raise_for_status()
    except requests.exceptions.RequestException as e:
        return None, f"Errore nel contattare il servizio di previsione produzione: {e}"

    dati_risposta = risposta.json()
    produzione_giornaliera = dati_risposta.get("result", {}).get("watt_hours_day")
    if not produzione_giornaliera:
        return None, None
    return produzione_giornaliera, None


def crea_tool_previsione_produzione(config_impianto: dict):
    """
    Factory che crea il tool di stima produzione futura, isolato per sessione utente.
    config_impianto: dict con chiavi 'lat', 'lon', 'kwp' (potenza impianto in kWp),
    opzionalmente 'dec' (inclinazione, default 30°) e 'az' (esposizione, default 0=sud).
    """
    @tool
    def previsione_produzione(giorni: int = 1) -> str:
        """
        Stima la produzione solare futura (in kWh) per le prossime ore/giorni, usando
        il servizio esterno Forecast.Solar basato sulla posizione geografica e sulla
        potenza dell'impianto fotovoltaico configurati dall'utente.

        A differenza del tool 'previsione_meteo' (che dice se ci sarà sole o pioggia),
        questo tool restituisce una stima numerica della produzione attesa in kWh.
        Usalo per domande tipo "quanto produrrò domani?" o per consigli proattivi che
        richiedono un numero (non solo "farà bel tempo").

        Args:
            giorni: quante giornate restituire A PARTIRE DA OGGI incluso (il primo giorno
                restituito è sempre oggi). 1 = solo oggi, 2 = oggi e domani (massimo
                disponibile: il servizio non fornisce stime oltre domani). Per rispondere
                su "domani" chiedi giorni=2, così la risposta includerà anche quella voce.
        """
        produzione_giornaliera, errore = _richiedi_produzione_forecast(config_impianto)
        if errore:
            return errore
        if not produzione_giornaliera:
            return "Il servizio di previsione produzione non ha restituito dati validi."

        oggi = datetime.now().date()
        righe = []
        for data_str, wh in sorted(produzione_giornaliera.items()):
            data_prevista = datetime.strptime(data_str, "%Y-%m-%d").date()
            diff = (data_prevista - oggi).days

            if diff < 0 or diff >= giorni:
                continue

            if diff == 0:
                etichetta = "oggi"
            elif diff == 1:
                etichetta = "domani"
            else:
                etichetta = f"tra {diff} giorni"

            kwh = wh / 1000
            righe.append(f"- {etichetta} ({data_str}): stima produzione {kwh:.1f} kWh")

        if not righe:
            return (
                f"Il servizio fornisce stime solo per oggi e domani "
                f"(hai chiesto una previsione a {giorni} giorni)."
            )

        lat = config_impianto["lat"]
        lon = config_impianto["lon"]
        kwp = config_impianto["kwp"]
        return (
            f"Stima produzione solare per l'impianto configurato ({kwp} kWp, "
            f"posizione lat={lat:.2f}, lon={lon:.2f}):\n" + "\n".join(righe)
        )

    return previsione_produzione


# ============================================================
# Gamification: missioni giornaliere
# ============================================================
def crea_tool_missione_giornaliera(dati: DatiEcoBot, config_impianto: dict):
    @tool
    def genera_missione_giornaliera() -> str:
        """
        Propone una missione/sfida di gamification per il giorno seguente, scelta tra tre tipi in base
        alla previsione meteo e al profilo di consumo dell'utente:
        - se domani è prevista una buona produzione solare: sposta consumo nella fascia di punta solare
          (badge 'Domatore di Sole');
        - altrimenti, se l'utente ha un picco di consumo serale pronunciato: riducilo (badge
          'Alleggerisci la Rete');
        - altrimenti: riduci il consumo totale giornaliero (badge 'Guardiano del Risparmio').
        Usalo quando l'utente chiede una sfida, una missione, un obiettivo giornaliero o genericamente
        qualcosa di "gamification" (es. "dammi una missione", "hai una sfida per me?").
        Se è già attiva una missione non ancora verificata, ne riporta lo stato invece di proporne
        una nuova.
        """
        if dati.produzione is None or dati.consumo is None:
            return "Nessun dato disponibile: chiama prima il tool pulisci_dati."

        missione_esistente = _carica_missione()
        if missione_esistente and missione_esistente.get("stato") == "in_corso":
            return _descrivi_missione_attiva(missione_esistente)

        produzione_giornaliera, errore = _richiedi_produzione_forecast(config_impianto)
        if errore:
            return f"Non riesco a proporre una missione al momento: {errore}"
        if not produzione_giornaliera:
            return "Il servizio di previsione produzione non ha restituito dati validi: non posso proporre una missione al momento."

        oggi = datetime.now().date()
        domani = oggi + timedelta(days=1)
        wh_domani = produzione_giornaliera.get(domani.isoformat())
        if wh_domani is None:
            return "Il servizio non fornisce ancora una stima per domani: riprova più tardi."

        kwh_domani = wh_domani / 1000
        media_storica_giornaliera = (
            dati.produzione.groupby(dati.produzione["timestamp"].dt.date)["kwh_produced"].sum().mean()
        )

        if kwh_domani >= media_storica_giornaliera:
            # Tipo 1: giornata soleggiata -> sposta consumo nella fascia di punta solare
            ora_inizio, ora_fine = _finestra_punta_solare(dati.produzione)
            baseline = _consumo_medio_finestra(dati.consumo, ora_inizio, ora_fine)
            target_kwh = baseline * (1 + PERCENTUALE_TARGET_SOLARE / 100)
            missione = {
                "tipo": "punta_solare",
                "data_target": domani.isoformat(),
                "ora_inizio": ora_inizio,
                "ora_fine": ora_fine,
                "direzione": "aumento",
                "percentuale_target": PERCENTUALE_TARGET_SOLARE,
                "badge": BADGE_MISSIONE_SOLARE,
                "consumo_baseline_kwh": round(baseline, 3),
                "stato": "in_corso",
            }
            _salva_missione(missione)
            return (
                f"Domani ({domani.isoformat()}) è prevista una buona giornata di sole "
                f"(~{kwh_domani:.1f} kWh stimati per il tuo impianto). Ecco la tua missione: sposta "
                f"almeno il {PERCENTUALE_TARGET_SOLARE}% del tuo consumo abituale nella fascia di punta "
                f"solare ({ora_inizio}:00-{ora_fine}:00, l'orario in cui il tuo impianto produce di più) "
                f"per sbloccare il badge '{BADGE_MISSIONE_SOLARE}'. Di solito in quella fascia consumi "
                f"circa {baseline:.2f} kWh al giorno: prova ad arrivare almeno a {target_kwh:.2f} kWh. "
                "Torna a chiedermi il risultato il giorno dopo, una volta caricati i dati aggiornati."
            )

        # Non è prevista una buona giornata di sole: valutiamo il profilo di consumo serale
        ora_inizio_serale, ora_fine_serale = _finestra_punta_consumo_serale(dati.consumo)
        baseline_serale = _consumo_medio_finestra(dati.consumo, ora_inizio_serale, ora_fine_serale)
        ore_finestra_serale = ora_fine_serale - ora_inizio_serale + 1
        media_oraria_serale = baseline_serale / ore_finestra_serale
        media_oraria_complessiva = dati.consumo["kwh_consumed"].mean()

        if media_oraria_serale >= media_oraria_complessiva * SOGLIA_PICCO_SERALE:
            # Tipo 2: picco di consumo serale pronunciato -> riducilo
            target_kwh = baseline_serale * (1 - PERCENTUALE_TARGET_RETE / 100)
            missione = {
                "tipo": "alleggerimento_rete",
                "data_target": domani.isoformat(),
                "ora_inizio": ora_inizio_serale,
                "ora_fine": ora_fine_serale,
                "direzione": "riduzione",
                "percentuale_target": PERCENTUALE_TARGET_RETE,
                "badge": BADGE_MISSIONE_RETE,
                "consumo_baseline_kwh": round(baseline_serale, 3),
                "stato": "in_corso",
            }
            _salva_missione(missione)
            return (
                f"Domani ({domani.isoformat()}) non è prevista una gran giornata di sole "
                f"(~{kwh_domani:.1f} kWh stimati), ma hai un picco di consumo piuttosto marcato la sera: "
                f"la tua missione è ridurre di almeno il {PERCENTUALE_TARGET_RETE}% il consumo nella "
                f"fascia serale di punta ({ora_inizio_serale}:00-{ora_fine_serale}:00) per sbloccare il "
                f"badge '{BADGE_MISSIONE_RETE}' e alleggerire la rete nelle ore di maggior richiesta. "
                f"Di solito in quella fascia consumi circa {baseline_serale:.2f} kWh: prova a scendere "
                f"sotto {target_kwh:.2f} kWh. Torna a chiedermi il risultato il giorno dopo, una volta "
                "caricati i dati aggiornati."
            )

        # Tipo 3: nessun picco particolare -> obiettivo generico di risparmio giornaliero
        baseline_giornaliera = dati.consumo.groupby(dati.consumo["timestamp"].dt.date)["kwh_consumed"].sum().mean()
        target_kwh = baseline_giornaliera * (1 - PERCENTUALE_TARGET_RISPARMIO / 100)
        missione = {
            "tipo": "risparmio_generale",
            "data_target": domani.isoformat(),
            "ora_inizio": None,
            "ora_fine": None,
            "direzione": "riduzione",
            "percentuale_target": PERCENTUALE_TARGET_RISPARMIO,
            "badge": BADGE_MISSIONE_RISPARMIO,
            "consumo_baseline_kwh": round(baseline_giornaliera, 3),
            "stato": "in_corso",
        }
        _salva_missione(missione)
        return (
            f"Domani ({domani.isoformat()}) non è prevista una gran giornata di sole "
            f"(~{kwh_domani:.1f} kWh stimati) e il tuo consumo è già abbastanza regolare durante il "
            f"giorno: la tua missione è ridurre il consumo totale di almeno il "
            f"{PERCENTUALE_TARGET_RISPARMIO}% rispetto alla tua media giornaliera, per sbloccare il "
            f"badge '{BADGE_MISSIONE_RISPARMIO}'. Di solito consumi circa {baseline_giornaliera:.2f} kWh "
            f"al giorno: prova a scendere sotto {target_kwh:.2f} kWh. Torna a chiedermi il risultato il "
            "giorno dopo, una volta caricati i dati aggiornati."
        )

    return genera_missione_giornaliera


def crea_tool_verifica_missione(dati: DatiEcoBot):
    @tool
    def verifica_missione() -> str:
        """
        Controlla se la missione/sfida attiva (proposta da genera_missione_giornaliera) è stata
        completata, confrontando il consumo reale nella data target (presente nei dati storici
        caricati) con l'obiettivo fissato al momento della missione. Usalo per domande tipo
        "ho completato la missione?", "com'è andata la sfida?", "ho sbloccato il badge?".
        """
        if dati.consumo is None:
            return "Nessun dato disponibile: chiama prima il tool pulisci_dati."

        missione = _carica_missione()
        if missione is None:
            return "Non c'è nessuna missione attiva al momento: chiedimi di proporne una."

        data_target = datetime.strptime(missione["data_target"], "%Y-%m-%d").date()
        consumo_reale = _consumo_giorno(dati.consumo, data_target, missione["ora_inizio"], missione["ora_fine"])

        if consumo_reale is None:
            return (
                f"I dati caricati non arrivano ancora al {missione['data_target']}: la missione è "
                "ancora in corso, torna a chiedermelo dopo aver caricato dati più recenti."
            )

        baseline = missione["consumo_baseline_kwh"]
        percentuale = missione["percentuale_target"]
        badge = missione["badge"]

        if missione["direzione"] == "aumento":
            target = baseline * (1 + percentuale / 100)
            riuscita = consumo_reale >= target
        else:
            target = baseline * (1 - percentuale / 100)
            riuscita = consumo_reale <= target

        _salva_missione(None)  # missione conclusa, si libera lo slot per la prossima

        if riuscita:
            badge_sbloccati = _carica_badge()
            badge_sbloccati.append(badge)
            _salva_badge(badge_sbloccati)
            return (
                f"🎉 Missione completata! Il {missione['data_target']} hai registrato "
                f"{consumo_reale:.2f} kWh (obiettivo: {target:.2f} kWh) — hai sbloccato il badge "
                f"'{badge}'. Badge sbloccati finora: {len(badge_sbloccati)}. Ottimo lavoro, questo si "
                "riflette positivamente sul tuo eco score settimanale: continua così!"
            )
        else:
            return (
                f"Missione non completata: il {missione['data_target']} hai registrato "
                f"{consumo_reale:.2f} kWh, l'obiettivo per sbloccare il badge '{badge}' era "
                f"{target:.2f} kWh. Niente di grave, puoi chiedermi una nuova missione quando vuoi "
                "e riprovare."
            )

    return verifica_missione


# ============================================================
# grafici
# ============================================================
def crea_tool_grafico(dati):
    @tool
    def genera_grafico(tipo: str, data_inizio: str = "", data_fine: str = "") -> str:
        """Genera e salva come immagine PNG un grafico per visualizzare l'andamento energetico.
        Da usare quando l'utente chiede esplicitamente un grafico, un andamento visivo o una curva.
        Args:
            tipo: tipo di grafico, uno tra:
                'produzione' (curva della produzione nel tempo),
                'consumo' (curva del consumo nel tempo),
                'confronto' (produzione e consumo sovrapposti nel tempo),
                'andamento_orario_produzione' (produzione media per ora del giorno),
                'andamento_orario_consumo' (consumo medio per ora del giorno).
            data_inizio: data di inizio in formato YYYY-MM-DD (opzionale).
            data_fine: data di fine in formato YYYY-MM-DD, inclusa (opzionale).
        """
        if dati.produzione is None or dati.consumo is None:
            return "Nessun dato disponibile: chiama prima il tool pulisci_dati."

        prod = _filtra_periodo(dati.produzione, data_inizio or None, data_fine or None)
        cons = _filtra_periodo(dati.consumo, data_inizio or None, data_fine or None)

        fig, ax = plt.subplots(figsize=(10, 5))

        if tipo == "produzione":
            ax.plot(prod["timestamp"], prod["kwh_produced"], color="#f5a623")
            ax.set_title("Produzione solare nel tempo")
            ax.set_xlabel("Data e ora")
            ax.set_ylabel("kWh prodotti")
            fig.autofmt_xdate()
        elif tipo == "consumo":
            ax.plot(cons["timestamp"], cons["kwh_consumed"], color="#4a90d9")
            ax.set_title("Consumo domestico nel tempo")
            ax.set_xlabel("Data e ora")
            ax.set_ylabel("kWh consumati")
            fig.autofmt_xdate()
        elif tipo == "confronto":
            ax.plot(prod["timestamp"], prod["kwh_produced"], label="Produzione", color="#f5a623")
            ax.plot(cons["timestamp"], cons["kwh_consumed"], label="Consumo", color="#4a90d9")
            ax.set_title("Produzione vs Consumo")
            ax.set_xlabel("Data e ora")
            ax.set_ylabel("kWh")
            ax.legend()
            fig.autofmt_xdate()
        elif tipo == "andamento_orario_produzione":
            medie = prod.groupby(prod["timestamp"].dt.hour)["kwh_produced"].mean()
            ax.bar(medie.index, medie.values, color="#f5a623")
            ax.set_title("Produzione media per ora del giorno")
            ax.set_xlabel("Ora del giorno")
            ax.set_ylabel("kWh medi")
            ax.set_xticks(range(0, 24, 2))
        elif tipo == "andamento_orario_consumo":
            medie = cons.groupby(cons["timestamp"].dt.hour)["kwh_consumed"].mean()
            ax.bar(medie.index, medie.values, color="#4a90d9")
            ax.set_title("Consumo medio per ora del giorno")
            ax.set_xlabel("Ora del giorno")
            ax.set_ylabel("kWh medi")
            ax.set_xticks(range(0, 24, 2))
        else:
            plt.close(fig)
            return (
                "Tipo di grafico non valido. Usa uno tra: 'produzione', 'consumo', 'confronto', "
                "'andamento_orario_produzione', 'andamento_orario_consumo'."
            )

        fig.tight_layout()
        timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        nome_file = f"{tipo}_{timestamp_file}.png"
        percorso = os.path.join(CARTELLA_GRAFICI, nome_file)
        fig.savefig(percorso, dpi=120)
        plt.close(fig)

        # Correzione versione precedente: il modello in automatico creava un percorso che portava a una nuova
        # pagina sul server streamlit in cui  mostrare il grafico mostrando pero una pagina senza niente
        # Soluzione: esplicitare al modello il comportamento per vincolarlo

        #Vecchia versione
        # dati.grafici_generati.append(percorso)
        # return f"Grafico generato e salvato in: {percorso}"

        dati.grafici_generati.append(percorso)
        return (
            "Grafico generato con successo. Verrà mostrato automaticamente all'utente subito "
            "dopo la tua risposta: non includere il percorso del file né un link ad esso nel "
            "messaggio finale, limitati a descrivere brevemente cosa mostra il grafico."
        )

    return genera_grafico


# ============================================================
# System prompt
# ============================================================
# NOTA: rispetto alla versione di test nel notebook, qui le regole 3 e 4 sono state riformulate
# dopo un bug riscontrato testando l'interfaccia Streamlit. La versione originale della regola 3
# ("prima enuncia il piano, poi agisci") poteva far terminare il turno all'agente subito dopo aver
# dichiarato l'intenzione, senza chiamare davvero i tool (il loop ReAct di LangGraph considera
# finale qualunque messaggio del modello privo di tool_calls). Non è un problema specifico di
# Streamlit, semplicemente non si è manifestato nei test nel notebook per la natura non
# deterministica del modello (temperature=0.3).
def costruisci_system_prompt():
    data_oggi = date.today().isoformat()
    return f"""Sei EcoBot, l'analista energetico virtuale di EcoGrid Solutions, una startup che gestisce
una comunità energetica rinnovabile (CER): piccoli produttori locali con impianti fotovoltaici condividono
l'energia in eccesso con famiglie della stessa area, riducendo bollette e impatto ambientale.

Parli con clienti finali che NON sono esperti di energia: dashboard tecniche con grafici di kWh e curve di
carico spesso li confondono. Il tuo compito è spiegare i loro dati energetici in modo semplice, chiaro e
gentile, ed essere proattivo nei consigli pratici (es. quando conviene accendere un elettrodomestico, se
vale la pena valutare una batteria di accumulo, ecc.) — mai dati grezzi senza interpretazione.

Data di oggi: {data_oggi}. Usala per calcolare correttamente periodi relativi menzionati dall'utente
(es. "la settimana scorsa", "questo mese", "luglio") prima di chiamare i tool, che richiedono date
precise in formato YYYY-MM-DD.

Hai a disposizione questi strumenti:
- pulisci_dati: carica e pulisce i CSV di produzione/consumo. Chiamalo per primo se i dati non sono
  ancora stati caricati (alcuni tool ti diranno esplicitamente di farlo se necessario).
- calcola_produzione / calcola_consumo: energia totale prodotta/consumata in un periodo.
- confronta_produzione_consumo: bilancio produzione vs consumo, ed evidenzia il giorno con più surplus.
- andamento_orario: fasce orarie di maggior produzione/consumo.
- individua_anomalie: eventuali guasti sensori o dati anomali rilevati durante la pulizia.
- genera_grafico: crea e salva un grafico PNG (produzione, consumo, confronto, andamento orario).
- calcola_kpi_energetici: indici di performance energetica in un periodo — autoconsumo, autosufficienza
  e percentuale di surplus ceduto alla rete/comunità. Usalo per domande più avanzate di "quanto ho
  prodotto/consumato", tipo "quanto sono autosufficiente?" o "quanto condivido con la comunità?".
- previsione_meteo: previsioni meteo (temperatura, radiazione solare attesa, probabilità di pioggia) per
  i prossimi giorni nella città dell'utente. I CSV contengono SOLO dati storici passati: per qualunque
  domanda sul futuro (previsioni, quando conviene fare qualcosa nei prossimi giorni) usa questo tool,
  non inventare un meteo futuro basandoti sui dati storici.
- previsione_produzione: stima numerica (in kWh) della produzione solare attesa per oggi/domani,
  basata sulla potenza dell'impianto fotovoltaico dell'utente. Usalo quando serve un NUMERO di
  produzione futura (es. "quanto produrrò domani?"), non solo un'indicazione generica di bel/brutto
  tempo — per quello usa previsione_meteo.
- co2_risparmiata: stima della CO2 evitata grazie all'energia rinnovabile prodotta in un periodo, utile
  per domande sul beneficio ambientale.
- punteggio_eco_settimanale: confronta l'indice di autoconsumo delle ultime due settimane per mostrare
  il progresso settimana su settimana, con un messaggio di incoraggiamento o di consiglio. Usalo per
  domande tipo "come sto andando questa settimana?" o "sono migliorato rispetto alla settimana scorsa?".
- genera_missione_giornaliera: propone una missione/sfida per il giorno seguente (sposta consumo nella
  fascia solare, riduci il picco serale, o riduci il consumo totale, a seconda del caso) con un badge
  da sbloccare. Usalo per richieste esplicite di una sfida, una missione o un obiettivo del giorno —
  è diverso da punteggio_eco_settimanale, che riguarda l'andamento generale, non una sfida specifica.
- verifica_missione: controlla se la missione attiva è stata completata, confrontando i dati aggiornati
  con l'obiettivo fissato. Usalo quando l'utente chiede l'esito della missione/sfida o se ha sbloccato
  un badge.

Regole di comportamento:
1. Basa sempre le tue risposte sui risultati reali dei tool: non inventare mai numeri.
2. Se un tool restituisce un messaggio di errore o un'istruzione (es. "chiama prima pulisci_dati"),
   segui quell'istruzione e riprova, invece di arrenderti o di inventare una risposta.
3. Per rispondere a una domanda che richiede analisi, chiama subito i tool necessari: non limitarti
   a descrivere cosa faresti, esegui davvero i passaggi finché non hai i risultati reali da mostrare.
4. Quando è utile, genera un grafico con genera_grafico. Nella risposta menziona solo che hai
   creato il grafico (es. "Ho preparato un grafico che mostra..."), senza includere il percorso
   del file né link ad esso: viene mostrato automaticamente sotto il messaggio.
5. Per domande su performance avanzate (autoconsumo, autosufficienza, surplus condiviso) usa
   calcola_kpi_energetici invece di limitarti a produzione/consumo totali.
6. Per domande sul futuro, su previsioni, o per consigli proattivi su quando conviene usare
   elettrodomestici nei prossimi giorni, usa previsione_meteo.
7. Quando è pertinente evidenziare il beneficio ambientale, usa co2_risparmiata.
8. Rispondi sempre in tono cordiale e alla portata di chi non ha competenze tecniche.
9. Quando i dati lo permettono, aggiungi un consiglio pratico e concreto, non generico — sfruttando
   se utile anche i KPI energetici o le previsioni meteo per renderlo più specifico.
10. Quando l'utente chiede una stima numerica di produzione futura (non solo se ci sarà sole),
    usa previsione_produzione, eventualmente insieme a previsione_meteo per il contesto.
11. Usa punteggio_eco_settimanale SOLO quando l'utente introduce l'argomento da zero, con una domanda
    generica come "come sto andando questa settimana?", "sono migliorato?" o "dammi il mio punteggio
    eco". Se invece la domanda è un follow-up che continua un'analisi specifica già in corso nella
    conversazione (es. "e rispetto alla settimana prima?" subito dopo un confronto produzione/consumo
    o un calcolo KPI), NON cambiare argomento: applica lo stesso tipo di analisi già richiesta in
    precedenza (stesso tool) al nuovo periodo indicato dall'utente.
12. Per richieste di una sfida, una missione o un obiettivo giornaliero con un badge da sbloccare, usa
    genera_missione_giornaliera; per sapere se è stata completata o se un badge è stato sbloccato, usa
    verifica_missione. Sono strumenti distinti da punteggio_eco_settimanale.
13. Non inventare mai se una missione è stata completata o se un badge è stato sbloccato: usa sempre
    verifica_missione per saperlo, e comunica onestamente anche se il risultato è che i dati non sono
    ancora disponibili per verificarla."""


# ============================================================
# Costruzione dell'agente per la current session
# ============================================================
@st.cache_resource
def crea_modello():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


def costruisci_agente(file_produzione, file_consumo, localita, config_impianto):
    dati = DatiEcoBot()
    tool_pulisci = crea_tool_pulizia(dati, file_produzione, file_consumo)
    tools = [
        tool_pulisci,
        crea_tool_produzione(dati),
        crea_tool_consumo(dati),
        crea_tool_confronto(dati),
        crea_tool_andamento_orario(dati),
        crea_tool_anomalie(dati),
        crea_tool_grafico(dati),
        crea_tool_kpi_energetici(dati),
        crea_tool_co2(dati),
        crea_tool_eco_score(dati),
        crea_tool_meteo(localita),
        crea_tool_previsione_produzione(config_impianto),
        crea_tool_missione_giornaliera(dati, config_impianto),
        crea_tool_verifica_missione(dati),
    ]
    agente = create_agent(
        model=crea_modello(),
        tools=tools,
        system_prompt=costruisci_system_prompt(),
        checkpointer=InMemorySaver(),
    )
    return dati, agente, tool_pulisci


# ============================================================
# sezione sidebar: upload dei due CSV + città per il meteo + potenza impianto
# ============================================================
with st.sidebar:
    st.header("Carica i tuoi dati")
    file_produzione = st.file_uploader("CSV produzione solare", type="csv")
    file_consumo = st.file_uploader("CSV consumo domestico", type="csv")
    città = st.text_input("La tua città (per le previsioni meteo)", value="Roma")
    kwp_impianto = st.number_input(
        "Potenza impianto fotovoltaico (kWp)",
        min_value=0.5, max_value=100.0, value=6.0, step=0.5,
        help="Usata per stimare quanto produrrà l'impianto nelle prossime ore/giorni "
             "(tool previsione_produzione). Inclinazione ed esposizione sono fissate a "
             "valori tipici (30°, esposizione sud) per semplicità.",
    )

if not file_produzione or not file_consumo:
    st.info("Carica entrambi i file CSV nella barra laterale per iniziare a chattare con EcoBot.")
    st.stop()

# la chiave include anche città e kWp: se cambiano, ricostruiamo l'agente con i nuovi parametri
chiave_agente = (
    file_produzione.name, file_produzione.size,
    file_consumo.name, file_consumo.size,
    città, kwp_impianto,
)

if st.session_state.get("chiave_agente") != chiave_agente:
    with st.spinner("Costruzione dell'agente e pulizia dei dati in corso..."):
        localita = geocodifica_citta(città) or LOCALITA_DEFAULT
        if localita is LOCALITA_DEFAULT and città.strip().lower() != "roma":
            st.sidebar.warning(f"Città '{città}' non trovata: uso Roma come località di default per il meteo.")
        config_impianto = {"lat": localita["lat"], "lon": localita["lon"], "kwp": kwp_impianto}
        dati, agente, tool_pulisci = costruisci_agente(file_produzione, file_consumo, localita, config_impianto)
        esito_pulizia = tool_pulisci.invoke({})
    st.session_state.chiave_agente = chiave_agente
    st.session_state.dati = dati
    st.session_state.agente = agente
    st.session_state.localita = localita
    st.session_state.thread_id = str(uuid.uuid4())  # id univoco di conversazione per questa sessione
    st.session_state.messaggi = []

dati = st.session_state.dati
agente = st.session_state.agente

with st.sidebar:
    st.success("Dati caricati e puliti ✅")
    st.caption(f"Meteo impostato su: {st.session_state.localita['nome']}")
    with st.expander("Dettaglio pulizia dati"):
        for riga in dati.log_pulizia:
            st.text(riga)

# ============================================================
# pannello KPI sempre visibile (calcolato sull'intero periodo disponibile)
# ============================================================
kpi = _calcola_kpi(dati.produzione, dati.consumo)
if kpi:
    st.subheader("Indicatori energetici")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Indice di autoconsumo", f"{kpi['indice_autoconsumo']:.0f}%")
    col2.metric("Indice di autosufficienza", f"{kpi['indice_autosufficienza']:.0f}%")
    col3.metric("Surplus condiviso", f"{kpi['surplus_ceduto']:.0f} kWh")
    col4.metric("CO2 evitata", f"{kpi['totale_prod'] * FATTORE_EMISSIONE_CO2_KWH:.0f} kg")
    st.divider()

# ============================================================
# banner eco score settimanale (gamification, sempre visibile)
# ============================================================
messaggio_eco_score = _messaggio_eco_score(dati)
if messaggio_eco_score:
    st.info(f"🏆 {messaggio_eco_score}")
    st.divider()

# ============================================================
# chat
# ============================================================
for messaggio in st.session_state.messaggi:
    with st.chat_message(messaggio["ruolo"]):
        st.markdown(messaggio["contenuto"])
        for percorso_immagine in messaggio.get("immagini", []):
            st.image(percorso_immagine)
        if messaggio.get("ragionamento"):
            with st.expander("Mostra ragionamento dell'agente"):
                for riga in messaggio["ragionamento"]:
                    st.markdown(riga)

domanda = st.chat_input("Fai una domanda a EcoBot sui tuoi dati energetici...")

if domanda:
    st.session_state.messaggi.append({"ruolo": "user", "contenuto": domanda})
    with st.chat_message("user"):
        st.markdown(domanda)

    with st.chat_message("assistant"):
        with st.spinner("EcoBot sta analizzando i dati..."):
            n_grafici_prima = len(dati.grafici_generati)
            config = {"configurable": {"thread_id": st.session_state.thread_id}}

            stato_precedente = agente.get_state(config)
            n_messaggi_precedenti = (
                len(stato_precedente.values.get("messages", [])) if stato_precedente.values else 0
            )

            risposta = agente.invoke({"messages": [{"role": "user", "content": domanda}]}, config=config)
            testo_risposta = risposta["messages"][-1].content
            nuovi_grafici = dati.grafici_generati[n_grafici_prima:]

            righe_ragionamento = []
            for messaggio in risposta["messages"][n_messaggi_precedenti:]:
                tipo = messaggio.__class__.__name__
                if tipo == "AIMessage":
                    for chiamata in getattr(messaggio, "tool_calls", []) or []:
                        righe_ragionamento.append(
                            f"🔧 Chiamata a **{chiamata['name']}** con argomenti `{chiamata['args']}`"
                        )
                elif tipo == "ToolMessage":
                    righe_ragionamento.append(f"↳ Risultato: {messaggio.content}")

        st.markdown(testo_risposta)
        for percorso_immagine in nuovi_grafici:
            st.image(percorso_immagine)
        if righe_ragionamento:
            with st.expander("Mostra ragionamento dell'agente"):
                for riga in righe_ragionamento:
                    st.markdown(riga)

    st.session_state.messaggi.append({
        "ruolo": "assistant",
        "contenuto": testo_risposta,
        "immagini": nuovi_grafici,
        "ragionamento": righe_ragionamento,
    })
