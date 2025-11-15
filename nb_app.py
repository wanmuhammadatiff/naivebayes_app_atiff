import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# ---------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Naive Bayes Classification App",
    layout="wide",
)

st.title("Naive Bayes Classification Web App")
st.write(
    "This app trains a Naive Bayes classifier on your dataset and "
    "shows performance metrics including an interactive confusion matrix."
)

# ---------------------------------------------------------
# 1. Upload / load dataset
# ---------------------------------------------------------
st.sidebar.header("Dataset Upload")

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)
st.success("Dataset loaded.")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Let user choose feature columns and target
st.sidebar.header("Features & Target")

all_columns = df.columns.tolist()

target_col = st.sidebar.selectbox("Target column (y)", all_columns, index=len(all_columns) - 1)
feature_cols = st.sidebar.multiselect(
    "Feature columns (X)",
    [c for c in all_columns if c != target_col],
    default=[c for c in all_columns if c != target_col],
)

if len(feature_cols) == 0:
    st.error("Please select at least one feature column.")
    st.stop()

X = df[feature_cols].values
y = df[target_col].values

# ---------------------------------------------------------
# 3. Train–test split & scaling
# ---------------------------------------------------------
st.sidebar.header("Train/Test Split")

test_size = st.sidebar.slider(
    "Test size (proportion for test set)",
    min_value=0.1,
    max_value=0.5,
    value=0.25,
    step=0.05,
)

random_state = st.sidebar.number_input(
    "Random state",
    value=42,
    step=1
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=int(random_state)
)

# Optional scaling (Naive Bayes doesn’t strictly need it, but it’s fine)
scale_features = st.sidebar.checkbox("Apply StandardScaler to features", value=True)

if scale_features:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

# ---------------------------------------------------------
# 4. Naive Bayes model 
# ---------------------------------------------------------
st.sidebar.header("Naive Bayes Parameters")

var_smoothing = st.sidebar.number_input(
    "var_smoothing (log10 scale, e.g. -9 for 1e-9)",
    value=-9.0,
    step=1.0,
)

nb_model = GaussianNB(var_smoothing=10 ** var_smoothing)
nb_model.fit(X_train, y_train)

# ---------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------
st.subheader("Model Performance")

y_pred = nb_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

st.metric("Accuracy", f"{acc:.3f}")

# --- interactive confusion matrix (table) ---
st.markdown("### Confusion Matrix (interactive table)")

# Build a labeled confusion matrix DataFrame
unique_labels = np.unique(np.concatenate([y_test, y_pred]))
cm_df = pd.DataFrame(
    cm,
    index=[f"Actual {lbl}" for lbl in unique_labels],
    columns=[f"Pred {lbl}" for lbl in unique_labels]
)

st.dataframe(cm_df)  # interactive table

# Classification report
st.markdown("### Classification Report")
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).T
st.dataframe(report_df.style.format("{:.3f}", na_rep="-"))

# ---------------------------------------------------------
# 6. Try a single prediction
# ---------------------------------------------------------
st.subheader("Try a New Prediction")

input_values = []
cols = st.columns(len(feature_cols)) if len(feature_cols) <= 4 else [st]

for i, col_name in enumerate(feature_cols):
    default_val = float(df[col_name].median()) if np.issubdtype(df[col_name].dtype, np.number) else 0.0
    with (cols[i] if len(feature_cols) <= 4 else st):
        val = st.number_input(f"{col_name}", value=default_val)
        input_values.append(val)

if st.button("Predict"):
    new_sample = np.array(input_values).reshape(1, -1)

    if scale_features:
        new_sample = scaler.transform(new_sample)

    pred_class = nb_model.predict(new_sample)[0]
    pred_proba = nb_model.predict_proba(new_sample)[0]

    st.write(f"**Predicted class:** `{pred_class}`")
    proba_df = pd.DataFrame(
        pred_proba.reshape(1, -1),
        columns=[f"Class {lbl}" for lbl in nb_model.classes_]
    )
    st.write("**Class probabilities:**")
    st.dataframe(proba_df)