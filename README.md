# Tema n 5 - Machine Learning per Processi Aziendali

## Smistamento Recensioni Hotel e Analisi Sentiment con Machine Learning

## Sebastiano Bizzi - matricola 0312301847

## Requisiti

- Python
- `pip`

## Installazione

Per facilitare l esecuzione è meglio usare un ambiente virtuale

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

Installare le dipendenze:

```bash
pip install -r requirements.txt
```

## Struttura del progetto

```
ProjectWork/
              app.py                  # Dashboard Streamlit
              requirements.txt        # Dipendenze Python
              data/
                  reviews.csv         # Dataset di training
              src/
                  train.py            # Script di training
                  predict.py          # Modulo di predizione
              models/                 # Modelli e vectorizer salvati (.pkl)
              output/                 # Metriche e grafici prodotti dal training
```

## Esecuzione

### Avvio della dashboard

I modelli pre-addestrati sono già presenti in `models/`, quindi la dashboard può essere avviata subito

```bash
streamlit run app.py
```

L'interfaccia è raggiungibile su `http://localhost:8501` e offre due modalità:

- **Analisi singola**: predizione di una recensione con parole più influenti
- **Batch analysis**: caricamento di un file CSV per analizzare più recensioni in blocco

### Riaddestramento dei modelli

Per rigenerare i modelli e le metriche a partire dal dataset

```bash
python src/train.py
```

i modelli vengono salvati in models/ e i grafici insieme al file metrics.json in output/


### Uso da terminale

Il modulo `src/predict.py` può essere importato per ottenere predizioni da codice Python

```python
from src.predict import predict_review

result = predict_review("Camera pulita e personale gentilissimo")
print(result)
```

## Dati

Il dataset di riferimento è `data/reviews.csv`