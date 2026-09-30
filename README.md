# EcoBot  -  Analista Energetico Virtuale (EcoGrid Solutions)

EcoBot è un agente AI data-driven che aiuta gli utenti di EcoGrid Solutions (una comunità energetica rinnovabile) a comprendere i propri dati di produzione solare e consumo domestico. Pulisce automaticamente i dati sporchi provenienti dai sensori IoT, risponde a domande di analisi in linguaggio naturale, calcola indicatori di performance energetica, genera grafici, consulta le previsioni meteo e la produzione solare futura per consigli proattivi, segnala i progressi settimanali dell'utente e propone missioni giornaliere con badge da sbloccare per rendere più coinvolgente il risparmio energetico.

## Modifiche rispetto al feedback del docente

Questa consegna risponde al feedback ricevuto in tre correzioni successive.

**Prima correzione**, due richieste:

1. **Tool basato su Forecast.Solar** per stimare la produzione fotovoltaica futura in kWh, non solo un'indicazione meteo generica → implementato come tool esterno `previsione_produzione`, con la potenza dell'impianto configurabile dall'utente in sidebar (dettagli nella sezione Architettura qui sotto).
2. **Gamification degli indicatori KPI** (eco score settimanale, messaggi di incoraggiamento tipo "sei salito al 75% di autoconsumo rispetto al 60% della settimana precedente") → implementato come tool conversazionale `punteggio_eco_settimanale` più un banner sempre visibile in Streamlit. Il confronto percentile con gli altri membri della community ("sei nel top X% della tua CER") non è stato incluso: avrebbe richiesto inventare dati di altri utenti non realmente presenti nel progetto, quindi si è preferito un confronto onesto sui soli dati reali dell'utente, settimana su settimana.

**Seconda correzione**: confermato corretto l'approccio del tool `previsione_produzione` (il giorno corrente è sempre incluso nella richiesta, e "domani" corrisponde a `giorni=2`). Sulla gamification è stato chiesto un passo ulteriore: trasformare i consigli in vere e proprie **missioni** con un obiettivo misurabile e un badge da sbloccare, verificate il giorno dopo sui dati aggiornati (es. "Sposta il 20% del tuo consumo abituale nelle ore di punta solare per sbloccare il badge 'Domatore di Sole'") → implementato con un sistema di tre missioni giornaliere diverse, descritto nella sezione "Missioni giornaliere e badge" più sotto.

**Terza correzione**: confermato che il progetto era sostanzialmente completo, con la sola richiesta di documentare meglio con degli screenshot il funzionamento delle funzionalità implementate nelle correzioni precedenti, in particolare i tool esterni e la gamification → aggiunti due screenshot supplementari agli screenshot della consegna, descritti nella sezione "Screenshot dell'agente in azione" più sotto.

## Architettura

Il progetto usa un'architettura **single-agent**, costruita con l'API moderna `create_agent` di LangChain 1.x (basata internamente su LangGraph), invece di un sistema multi-agente.

**Perché single-agent:** il dominio del problema è ben definito e circoscritto (analisi dei dati energetici di un'unica azienda produttrice e un'unica famiglia consumatrice), e un singolo set di tool ben progettato copre l'intero perimetro funzionale richiesto. Introdurre più agenti specializzati avrebbe aggiunto complessità di coordinamento senza un reale beneficio in termini di qualità delle risposte.

**Componenti principali:**
- **14 tool**, ciascuno una funzione Python decorata con `@tool`:
  - `pulisci_dati`, `calcola_produzione`, `calcola_consumo`, `confronta_produzione_consumo`, `andamento_orario`, `individua_anomalie`, `genera_grafico` — analisi di base sui dati caricati dall'utente.
  - `calcola_kpi_energetici` — indicatori di performance energetica avanzati: indice di autoconsumo, indice di autosufficienza, percentuale di surplus condiviso con la comunità.
  - `co2_risparmiata` — stima della CO2 evitata grazie all'energia rinnovabile prodotta (fattore di emissione medio della rete italiana, fonte ISPRA).
  - `punteggio_eco_settimanale` — confronta l'indice di autoconsumo delle ultime due settimane e restituisce un messaggio di incoraggiamento o di consiglio ("eco score" settimanale).
  - `previsione_meteo` — **tool esterno**: interroga l'API gratuita Open-Meteo per temperatura, radiazione solare attesa e probabilità di pioggia nei prossimi giorni, geolocalizzato sulla città indicata dall'utente.
  - `previsione_produzione` — **tool esterno**: interroga l'API gratuita Forecast.Solar per stimare in kWh la produzione solare attesa oggi/domani, in base alla posizione geografica e alla potenza dell'impianto fotovoltaico indicata dall'utente. A differenza di `previsione_meteo` (che descrive il tempo atteso), restituisce un numero utilizzabile direttamente per decidere quando conviene usare un elettrodomestico. Per semplicità l'inclinazione e l'esposizione dei pannelli sono fissate a valori tipici (30°, esposizione sud); solo la potenza dell'impianto è configurabile dall'utente.
    Nota: il tier gratuito di Forecast.Solar (nessuna chiave richiesta) ha un limite di 12 richieste/ora legato all'IP. Se il limite viene superato, il tool restituisce un messaggio di errore descrittivo invece di un numero inventato, e l'agente risponde comunque in modo utile basandosi sulle previsioni meteo disponibili — comportamento verificato e voluto, non un guasto.
  - `genera_missione_giornaliera` — propone una missione/sfida di gamification per il giorno seguente, con un badge da sbloccare (dettagli nella sezione "Missioni giornaliere e badge" più sotto).
  - `verifica_missione` — controlla se la missione attiva è stata completata, confrontando i dati aggiornati con l'obiettivo fissato, e sblocca il badge in caso di successo.
- I tool che operano sui dati condividono un data store comune (`DatiEcoBot`), passato tramite pattern factory (`crea_tool_x(dati)`); i tool `previsione_meteo`, `previsione_produzione`, `genera_missione_giornaliera` e `verifica_missione` ricevono allo stesso modo, dall'esterno, la località/configurazione dell'impianto scelte dall'utente, invece di usare costanti globali, per restare isolati correttamente tra sessioni utente diverse in Streamlit.
- **Memoria conversazionale**: checkpointer `InMemorySaver` di LangGraph, con un `thread_id` univoco per ogni sessione utente in Streamlit — permette all'agente di gestire domande di follow-up (es. "E rispetto alla settimana prima?") ricordando il contesto della conversazione.
- **System prompt strutturato**: definisce ruolo, tono (semplice, cordiale, per utenti non esperti), contesto operativo (EcoGrid Solutions, CER) e regole di comportamento esplicite su quando usare ciascun tool.
- **Gestione errori/auto-correzione**: i tool restituiscono messaggi di errore descrittivi e istruzioni (es. "chiama prima il tool pulisci_dati") quando i dati non sono ancora pronti; il system prompt istruisce l'agente a seguire queste istruzioni e ritentare invece di arrendersi o inventare una risposta.
- **Pannello KPI sempre visibile**: in Streamlit, una riga di metriche (autoconsumo, autosufficienza, surplus condiviso, CO2 evitata) è mostrata sopra la chat non appena i dati vengono caricati, oltre a essere disponibile anche tramite conversazione.
- **Banner eco score settimanale**: sempre visibile sopra la chat, mostra il confronto dell'indice di autoconsumo tra la settimana corrente e quella precedente, con un messaggio di incoraggiamento o di consiglio pratico. La stessa logica di calcolo è riusata anche dal tool conversazionale `punteggio_eco_settimanale`, per non duplicarla in due punti diversi.
  Nota: il confronto è calcolato sui dati reali dell'utente (non su un dataset multi-utente reale); per questo motivo non è stato incluso un confronto percentile con gli altri membri della community ("sei nel top X% della tua CER"), che avrebbe richiesto di inventare dati di altri utenti non realmente presenti nel progetto.
  Nota implementativa: nel notebook la funzione di supporto `_indice_autoconsumo_periodo` calcola l'autoconsumo in modo autonomo (i tool del notebook erano stati scritti come standalone, senza un helper `_calcola_kpi` condiviso), mentre in `app.py` la stessa funzione riusa l'helper `_calcola_kpi` già impiegato anche dal pannello KPI. Il risultato numerico è identico in entrambi i casi: cambia solo quanto viene riusato il codice tra le due versioni.

## Missioni giornaliere e badge

Per rendere concreta la gamification richiesta, ogni giorno l'utente può chiedere una missione: l'agente ne sceglie una tra tre tipi, in base alla previsione per il giorno seguente e al profilo di consumo storico dell'utente (non è mai una scelta casuale):

1. **Domatore di Sole** — se domani è prevista una buona produzione solare (previsione superiore alla media storica dell'impianto): sposta almeno il 20% del consumo abituale nella fascia oraria in cui l'impianto produce di più (individuata dinamicamente dai dati dell'utente, non fissa).
2. **Alleggerisci la Rete** — se domani non è prevista una gran giornata di sole ma l'utente ha un picco di consumo serale pronunciato: riduci di almeno il 15% il consumo nella fascia serale di punta, per alleggerire la rete nelle ore di maggior richiesta collettiva.
3. **Guardiano del Risparmio** — se nessuno dei due casi precedenti si applica: riduci il consumo totale del giorno di almeno il 15% rispetto alla propria media giornaliera.

Il giorno dopo, una volta caricati i CSV aggiornati che includono la data target, l'utente può chiedere l'esito ("ho completato la missione?"): l'agente confronta il consumo reale con l'obiettivo fissato al momento della missione e, in caso di successo, sblocca il badge corrispondente. Se i dati non arrivano ancora alla data target, lo dice onestamente invece di indovinare. La missione attiva e l'elenco dei badge sbloccati sono salvati in due piccoli file JSON locali (`missione_attiva.json`, `badge_sbloccati.json`) creati automaticamente nella cartella del progetto, così sopravvivono a un riavvio di Streamlit tra un giorno e l'altro.

**Nota di trasparenza sulla verifica con il dataset sintetico:** il meccanismo di verifica è pensato per funzionare correttamente con dati IoT reali e continuativi, dove il consumo del giorno successivo riflette davvero il comportamento dell'utente. Lo script di generazione dati fornito dalla consegna, però, crea un dataset sintetico *casuale* ogni volta che viene eseguito (sempre gli ultimi 30 giorni, generati da zero), non un'estensione incrementale dei giorni precedenti. Con questo dataset dimostrativo, quindi, l'esito della verifica riflette il nuovo campione casuale generato per quel giorno, non un vero tracciamento del comportamento dell'utente: la stessa logica, applicata a dati reali continuativi, funzionerebbe correttamente.

## Istruzioni di esecuzione

1. Assicurarsi di avere Python 3.x con Jupyter Notebook (consigliato: Anaconda).
2. Nella cartella del progetto, creare un file `.env` con la propria chiave API OpenAI:
```
   OPENAI_API_KEY=sk-...
```
3. Installare le dipendenze:
```
   pip install -r requirements.txt
```
4. Aprire `agent.ipynb` in Jupyter Notebook.
5. Eseguire la cella nella sezione "Generazione dataset sintetico" (la prima del notebook) per creare i due CSV di esempio, `solar_production_raw.csv` e `household_consumption.csv`, con dati simulati e alcune anomalie tipiche dei sensori IoT.
6. Eseguire in ordine le celle della sezione successiva ("costruzione dell'agente") per vedere setup, tool, reasoning loop e i test dell'agente direttamente nel notebook.
7. Eseguire la cella `%%writefile app.py` (sezione "Streamlit") — scrive/aggiorna il file `app.py` con la logica dell'agente validata sopra.
8. Eseguire l'ultima cella del notebook (`!streamlit run app.py`) — l'app si apre automaticamente nel browser su `http://localhost:8501`.
9. Nella barra laterale, caricare i due file CSV generati al punto 5 (`solar_production_raw.csv` per la produzione, `household_consumption.csv` per il consumo), indicare la propria città (usata per le previsioni meteo; il default è Roma) e la potenza del proprio impianto fotovoltaico in kWp (usata per la stima della produzione futura; il default è 6 kWp).
10. Una volta caricati e puliti i dati, iniziare a chattare con EcoBot nella chat interattiva. I file di persistenza delle missioni (`missione_attiva.json`, `badge_sbloccati.json`) vengono creati automaticamente nella cartella del progetto al bisogno: non serve alcuna azione manuale.

## Esempi di utilizzo

Domande testate con successo durante lo sviluppo:

- *"C'è stato un guasto ai sensori?"* → l'agente individua le anomalie rilevate durante la pulizia dei dati.
- *"Quanto è stato prodotto nel mese di luglio?"* → calcolo dell'energia totale prodotta nel periodo.
- *"Confrontami produzione e consumo delle ultime due settimane, e fammi un grafico"* → confronto testuale + grafico generato e mostrato direttamente in chat.
- *"E rispetto alla settimana prima?"* → domanda di follow-up che dimostra la memoria conversazionale dell'agente (nessun contesto ripetuto dall'utente).
- *"In quali ore del giorno consumo di più?"* → andamento orario medio dei consumi, con individuazione della fascia di picco.
- *"Quanto sono autosufficiente, e quanto condivido con la comunità?"* → indici di autoconsumo, autosufficienza e surplus condiviso.
- *"Conviene usare la lavatrice domani pomeriggio?"* → l'agente consulta le previsioni meteo reali (radiazione solare attesa, probabilità di pioggia) e dà un consiglio proattivo basato sui dati.
- *"Quanta CO2 ho risparmiato producendo energia pulita questo mese?"* → stima dell'impatto ambientale evitato.
- *"Quanto produrrò domani?"* → stima numerica in kWh della produzione futura, calcolata dal tool esterno `previsione_produzione` in base a impianto e posizione dell'utente.
- *"Come sto andando questa settimana?"* → confronto dell'indice di autoconsumo con la settimana precedente, con un messaggio di incoraggiamento o di consiglio pratico (visibile anche nel banner sempre presente sopra la chat).
- *"Dammi una missione per domani"* → propone una missione di gamification (una delle tre tipologie, in base a previsione e profilo di consumo) con un obiettivo misurabile in kWh e un badge da sbloccare.
- *"Ho completato la missione?"* → verifica l'esito della missione attiva confrontando i dati aggiornati con l'obiettivo; se i dati non includono ancora la data target, lo dice onestamente invece di indovinare.

Per ogni risposta che ha richiesto l'uso di tool, l'interfaccia Streamlit mostra anche un pannello "Mostra ragionamento dell'agente" con il dettaglio dei tool effettivamente chiamati e dei relativi risultati.

## Screenshot dell'agente in azione

Oltre allo screenshot richiesto dalla consegna (`streamlit_screenshot.png`), sono inclusi due screenshot supplementari per documentare più chiaramente i tool esterni e la gamification, le funzionalità aggiunte nelle correzioni successive:

- `streamlit_screenshot.png` — gamification: l'agente propone una missione giornaliera in risposta a *"Dammi una missione per domani"*.
- `streamlit_screenshot_2.png` — tool esterno `previsione_meteo`: consiglio proattivo in risposta a *"Conviene usare la lavatrice domani pomeriggio?"*.
- `streamlit_screenshot_3.png` — tool esterno `previsione_produzione` (Forecast.Solar): stima numerica in kWh in risposta a *"Quanto produrrò domani?"*.