# Modulo di predizione: carica i modelli e predice reparto e sentiment

import os
import re
import joblib

# Carico modelli e vectorizer una sola volta all'import del modulo
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(root, "models")

dept_model = joblib.load(os.path.join(models_dir, "department_model.pkl"))
sent_model = joblib.load(os.path.join(models_dir, "sentiment_model.pkl"))
dept_vectorizer = joblib.load(os.path.join(models_dir, "tfidf_dept.pkl"))
sent_vectorizer = joblib.load(os.path.join(models_dir, "tfidf_sent.pkl"))


def preprocess(text):
    # Rimozione punteggiatura (stesso preprocessing del training)
    return re.sub(r"[^\w\s]", "", text)


def top_influential_words(X, model, vectorizer, predicted_class, k=5):
    # Restituisce le k parole che hanno contribuito di piu alla
    # predizione della classe scelta. Contributo = tfidf_parola * coefficiente_classe
    feature_names = vectorizer.get_feature_names_out()
    x = X.toarray()[0]

    if model.coef_.shape[0] == 1:
        # Classificatore binario: coef_ rappresenta la classe "positive"
        if predicted_class == model.classes_[1]:
            coefs = model.coef_[0]
        else:
            coefs = -model.coef_[0]
    else:
        # Classificatore multi-classe (one-vs-rest)
        class_idx = list(model.classes_).index(predicted_class)
        coefs = model.coef_[class_idx]

    contributions = x * coefs
    top_idx = contributions.argsort()[::-1][:k]
    return [(feature_names[i], float(contributions[i]))
            for i in top_idx if contributions[i] > 0]

# Funzione per la predizione di una singola recensione
def predict(text):
    processed = preprocess(text)
    X_dept = dept_vectorizer.transform([processed])
    X_sent = sent_vectorizer.transform([processed])

    dept = dept_model.predict(X_dept)[0]
    dept_proba = dept_model.predict_proba(X_dept)[0]

    sent = sent_model.predict(X_sent)[0]
    sent_proba = sent_model.predict_proba(X_sent)[0]

    return {
        "department": dept,
        "sentiment": sent,
        "dept_confidence": max(dept_proba),
        "sent_confidence": max(sent_proba),
        "dept_probabilities": dict(zip(dept_model.classes_, dept_proba)),
        "sent_probabilities": dict(zip(sent_model.classes_, sent_proba)),
        "top_dept_words": top_influential_words(X_dept, dept_model, dept_vectorizer, dept),
        "top_sent_words": top_influential_words(X_sent, sent_model, sent_vectorizer, sent),
    }

# Funzione per la predizione di un batch di recensioni 
def predict_batch(df):
    texts = (df["title"] + " " + df["body"]).apply(preprocess)
    X_dept = dept_vectorizer.transform(texts)
    X_sent = sent_vectorizer.transform(texts)

    result = df.copy()
    result["department_pred"] = dept_model.predict(X_dept)
    result["sentiment_pred"] = sent_model.predict(X_sent)
    result["dept_confidence"] = [max(p) for p in dept_model.predict_proba(X_dept)]
    result["sent_confidence"] = [max(p) for p in sent_model.predict_proba(X_sent)]

    return result
