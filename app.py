import streamlit as st
import requests
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.drift_analysis import compare_datasets

# --- CONFIGURATION & CONSTANTS ---
SERVING_API_URL = os.getenv("SERVING_API_URL", "http://serving_api:8000")
PREFECT_API_URL = os.getenv("PREFECT_API_URL", "http://orchestrator:4200/api")
DATA_DIR = 'data'
LATEST_METRICS = 'models/latest_metrics.json'
PREDICTIONS_PATH = 'data/predictions.csv'

st.set_page_config(
    page_title="Process MLOps Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLING (CSS) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #F8FAFC 0%, #E2E8F0 100%); }
    h1, h2, h3 { color: #0F172A !important; font-weight: 700; }
    div[data-testid="metric-container"] {
        background-color: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.5);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

# --- UTILS ---
def trigger_orchestrator(flow_name, params=None):
    """Placeholder for triggering Prefect flows via HTTP"""
    # In a real setup, we would call the Prefect REST API to create a flow run.
    # For this MVP, we will simulate the trigger which runs in the orchestrator container.
    try:
        # Simplified: We assume a generic 'trigger' endpoint or similar
        # For now, let's log the attempt. 
        st.toast(f"🚀 Triggering FLOW: {flow_name}", icon="⚡")
        # In a real containerized env, this would be a POST to Prefect
        return True
    except Exception as e:
        st.error(f"Error triggering orchestrator: {e}")
        return False

def get_predictions_from_api(data_path):
    try:
        df = pd.read_csv(data_path).sample(20).to_dict('records')
        response = requests.post(f"{SERVING_API_URL}/predict", json=df)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.text}")
            return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/brain-3.png", width=60)
    st.title("Process MLOps")
    st.markdown("---")
    
    st.subheader("📦 Data Governance")
    if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
    archivos_disponibles = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.csv') and not f.startswith('predictions')])
    
    dataset_seleccionado = st.selectbox("Dataset Activo", options=archivos_disponibles, index=len(archivos_disponibles)-1 if archivos_disponibles else None)
    
    if st.button("Generar Datos (Syntetic)", type="primary"):
        trigger_orchestrator("generate_data")
        # For MVP, we still run locally to see immediate results in shared volume
        import subprocess
        subprocess.run(["python", "-m", "src.sintetic_gen"])
        st.rerun()

    st.markdown("---")
    st.subheader("⚡ Pipelines")
    
    if st.button("🚀 Entrenar Modelo", disabled=not dataset_seleccionado):
        trigger_orchestrator("train_model", {"data": dataset_seleccionado})
        import subprocess
        subprocess.run(["python", "-m", "src.train_pipeline", "--data", dataset_seleccionado])
        st.rerun()

    if st.button("🔮 Inferencia API", disabled=not dataset_seleccionado):
        with st.spinner("Llamando a Serving API..."):
            results = get_predictions_from_api(os.path.join(DATA_DIR, dataset_seleccionado))
            if results:
                res_df = pd.DataFrame(results)
                res_df.to_csv(PREDICTIONS_PATH, index=False)
                st.success("Resultados recibidos de la API.")
                st.rerun()
            
    st.markdown("---")
    st.caption("Evolved Architecture v3.0")

# --- MAIN ---
st.title("📊 Employee Attrition Dashboard")
tab1, tab2, tab3 = st.tabs(["📈 Model Overview", "🕵️ Drift Analysis", "🔍 Inference Inspector"])

with tab1:
    st.markdown("### Rendimiento del Modelo")
    
    # List available specific metrics
    metric_files = sorted([f for f in os.listdir('models') if f.startswith('metrics_') and f.endswith('.json')], reverse=True)
    
    col_sel1, col_sel2 = st.columns([1, 1])
    with col_sel1:
        selected_model = st.selectbox("Comparar Modelo Histórico:", ["Ninguno"] + metric_files)
    
    if os.path.exists(LATEST_METRICS):
        with open(LATEST_METRICS, 'r') as f: latest = json.load(f)
        
        # Display Logic
        if selected_model != "Ninguno":
            with open(os.path.join('models', selected_model), 'r') as f: historical = json.load(f)
            
            st.markdown(f"#### Comparativa: `Latest` vs `{selected_model.replace('metrics_', '').replace('.json', '')}`")
            c1, c2, c3, c4 = st.columns(4)
            
            def get_f1(m): return m.get('classification_report', {}).get('1', {}).get('f1-score', 0)
            def get_prec(m): return m.get('classification_report', {}).get('1', {}).get('precision', 0)
            def get_rec(m): return m.get('classification_report', {}).get('1', {}).get('recall', 0)

            c1.metric("ROC-AUC", f"{latest.get('roc_auc', 0):.3f}", f"{latest.get('roc_auc', 0) - historical.get('roc_auc', 0):.3f}")
            c2.metric("F1-Score", f"{get_f1(latest):.3f}", f"{get_f1(latest) - get_f1(historical):.3f}")
            c3.metric("Precision", f"{get_prec(latest):.3f}", f"{get_prec(latest) - get_prec(historical):.3f}")
            c4.metric("Recall", f"{get_rec(latest):.3f}", f"{get_rec(latest) - get_rec(historical):.3f}")
        else:
            st.markdown(f"#### Modelo Actual (`{latest.get('dataset_version', 'N/A')}`)")
            c1, c2, c3, c4 = st.columns(4)
            report = latest.get('classification_report', {})
            c1.metric("ROC-AUC", f"{latest.get('roc_auc', 0):.4f}")
            c2.metric("F1-Score", f"{report.get('1', {}).get('f1-score', 0):.4f}")
            c3.metric("Precision", f"{report.get('1', {}).get('precision', 0):.4f}")
            c4.metric("Recall", f"{report.get('1', {}).get('recall', 0):.4f}")
    else:
        st.info("Inicia el proceso generando datos y entrenando el modelo.")

with tab2:
    if len(archivos_disponibles) >= 2:
        v1 = st.selectbox("Versión Base", archivos_disponibles, index=0)
        v2 = st.selectbox("Versión Actual", archivos_disponibles, index=len(archivos_disponibles)-1)
        
        if st.button("Analizar Drift"):
            path_old = os.path.join(DATA_DIR, v1)
            path_new = os.path.join(DATA_DIR, v2)
            df_old = pd.read_csv(path_old)
            df_new = pd.read_csv(path_new)
            
            drift = compare_datasets(path_old, path_new)
            
            # KPI Metrics
            st.markdown("#### Desviaciones Clave")
            cols = st.columns(4)
            for i, (k, v) in enumerate(drift.items()):
                if v and v['type'] == 'numeric':
                    cols[i % 4].metric(k, f"{v['mean_new']:.1f}", f"{v['drift_pct']:.2f}%")
            
            # Visualizations
            st.markdown("---")
            st.markdown("#### Comparativa de Distribuciones")
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Monthly Income Chart
            sns.kdeplot(df_old['MonthlyIncome'], label=v1, ax=axes[0], fill=True, alpha=0.3)
            sns.kdeplot(df_new['MonthlyIncome'], label=v2, ax=axes[0], fill=True, alpha=0.3)
            axes[0].set_title("Shift en MonthlyIncome")
            axes[0].legend()
            
            # Job Satisfaction Chart (Categorical representation)
            if 'JobSatisfaction' in df_old.columns and 'JobSatisfaction' in df_new.columns:
                sat_old = df_old['JobSatisfaction'].value_counts(normalize=True).sort_index()
                sat_new = df_new['JobSatisfaction'].value_counts(normalize=True).sort_index()
                
                compare_sat = pd.DataFrame({'Base': sat_old, 'Actual': sat_new})
                compare_sat.plot(kind='bar', ax=axes[1])
                axes[1].set_title("Shift en Job Satisfaction")
                axes[1].set_ylabel("Proporción")
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Detailed Drift Table
            st.markdown("#### Detalle Por Centiles")
            stats_df = pd.DataFrame({
                'Métrica': ['Mean', 'Std', 'Min', '50%', 'Max'],
                v1: [df_old['MonthlyIncome'].mean(), df_old['MonthlyIncome'].std(), df_old['MonthlyIncome'].min(), df_old['MonthlyIncome'].median(), df_old['MonthlyIncome'].max()],
                v2: [df_new['MonthlyIncome'].mean(), df_new['MonthlyIncome'].std(), df_new['MonthlyIncome'].min(), df_new['MonthlyIncome'].median(), df_new['MonthlyIncome'].max()]
            })
            st.table(stats_df)
    else:
        st.warning("Se requieren 2 datasets para el análisis.")

with tab3:
    if os.path.exists(PREDICTIONS_PATH):
        df_preds = pd.read_csv(PREDICTIONS_PATH)
        # Fix: handle both 'probability' and 'Churn_Probability' column names
        prob_col = 'probability' if 'probability' in df_preds.columns else 'Churn_Probability'
        
        if prob_col in df_preds.columns:
            st.dataframe(df_preds.style.background_gradient(subset=[prob_col], cmap='RdYlGn_r'), use_container_width=True)
        else:
            st.dataframe(df_preds, use_container_width=True)
    else:
        st.info("Lanza una inferencia desde el panel lateral para ver resultados de la API.")
