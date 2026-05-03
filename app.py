#Dashboard Streamlit 

import streamlit as st
import pandas as pd
from datetime import datetime
from src.predict import predict, predict_batch
# Configurazione della pagina   
st.set_page_config(page_title="Smistamento Recensioni Hotel", layout="wide")

st.title("Smistamento Recensioni Hotel")
st.markdown("Classificazione automatica per **reparto** e **sentiment** con Machine Learning")

tab1, tab2 = st.tabs(["Analisi Singola", "Analisi Batch"])

# Tab analisi singola 
with tab1:
    st.subheader("Inserisci una recensione")

    review_text = st.text_area(
        "Scrivi o incolla qui il testo della recensione:",
        height=150,
        placeholder="Es: La camera era pulitissima, lenzuola fresche e bagno impeccabile.",
    )
# Bottone per analizzare la recensione
    if st.button("Analizza", type="primary"):
        if review_text.strip():
            result = predict(review_text)

            col1, col2 = st.columns(2)

            with col1:
                st.metric(label="Reparto consigliato", value=result["department"])
                st.caption(f"Confidenza: {result['dept_confidence']:.1%}")
                st.progress(result["dept_confidence"])

                st.markdown("**Distribuzione probabilita' per reparto**")
                dept_probs = pd.DataFrame(
                    [{"reparto": k, "probabilita": v}
                     for k, v in sorted(result["dept_probabilities"].items(),
                                        key=lambda x: -x[1])]
                )
                dept_probs["probabilita"] = dept_probs["probabilita"].apply(lambda v: f"{v:.1%}")
                st.dataframe(dept_probs, hide_index=True, use_container_width=True)

                st.markdown("**Top 5 parole influenti**")
                if result["top_dept_words"]:
                    df_dept = pd.DataFrame(
                        result["top_dept_words"], columns=["parola", "contributo"]
                    )
                    df_dept["contributo"] = df_dept["contributo"].round(3)
                    st.dataframe(df_dept, hide_index=True, use_container_width=True)
                else:
                    st.caption("Nessuna parola con contributo positivo.")

            with col2:
                st.metric(label="Sentiment", value=result["sentiment"])
                st.caption(f"Confidenza: {result['sent_confidence']:.1%}")
                st.progress(result["sent_confidence"])

                st.markdown("**Distribuzione probabilita' sentiment**")
                sent_probs = pd.DataFrame(
                    [{"classe": k, "probabilita": v}
                     for k, v in sorted(result["sent_probabilities"].items(),
                                        key=lambda x: -x[1])]
                )
                sent_probs["probabilita"] = sent_probs["probabilita"].apply(lambda v: f"{v:.1%}")
                st.dataframe(sent_probs, hide_index=True, use_container_width=True)

                st.markdown("**Top 5 parole influenti**")
                if result["top_sent_words"]:
                    df_sent = pd.DataFrame(
                        result["top_sent_words"], columns=["parola", "contributo"]
                    )
                    df_sent["contributo"] = df_sent["contributo"].round(3)
                    st.dataframe(df_sent, hide_index=True, use_container_width=True)
                else:
                    st.caption("Nessuna parola con contributo positivo.")
        else:
            st.warning("Inserisci il testo di una recensione prima di analizzare.")

# Tab analisi batch 
with tab2:
    st.subheader("Carica un file CSV")
    st.markdown("Il CSV deve contenere le colonne **title** e **body**.")
# Caricamento del file CSV
    uploaded_file = st.file_uploader("Scegli un file CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if "title" not in df.columns or "body" not in df.columns:
            st.error("Il CSV deve contenere le colonne 'title' e 'body'.")
        else:
            st.write(f"Caricate **{len(df)}** recensioni.")
# Bottone per analizzare il batch
            if st.button("Analizza batch", type="primary"):
                with st.spinner("Analisi in corso..."):
                    results = predict_batch(df)

                st.success(f"Analisi completata per {len(results)} recensioni.")

                st.dataframe(
                    results[["title", "body", "department_pred", "sentiment_pred",
                             "dept_confidence", "sent_confidence"]],
                    use_container_width=True,
                )

# Export con timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_output = results.to_csv(index=False)

                st.download_button(
                    label="Scarica risultati CSV",
                    data=csv_output,
                    file_name=f"predizioni_{timestamp}.csv",
                    mime="text/csv",
                )
