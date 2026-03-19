import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="HITL Decision Simulation", layout="wide")

# =========================================================
# Helpers
# =========================================================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def evaluate(preds, truth):
    preds, truth = np.array(preds), np.array(truth)
    tp = int(np.sum((preds == 1) & (truth == 1)))
    tn = int(np.sum((preds == 0) & (truth == 0)))
    fp = int(np.sum((preds == 1) & (truth == 0)))
    fn = int(np.sum((preds == 0) & (truth == 1)))
    acc = np.mean(preds == truth)
    return {
        "Accuracy": round(float(acc), 3),
        "False Positives": fp,
        "False Negatives": fn
    }

def expected_cost(preds, truth, review_mask=None, fn_cost=10, fp_cost=2, review_cost=1):
    preds, truth = np.array(preds), np.array(truth)
    fp = np.sum((preds == 1) & (truth == 0))
    fn = np.sum((preds == 0) & (truth == 1))
    reviews = int(np.sum(review_mask)) if review_mask is not None else 0
    return int(fn * fn_cost + fp * fp_cost + reviews * review_cost)

def format_case_summary(row):
    return f"""
**Case {int(row['case_id'])}**
- Amount: **${row['amount']:,.2f}** | Age: **{int(row['age'])}** | Account Age: **{int(row['account_age_days'])} days**
- Model Risk Score: **{row['model_score']:.3f}**
- **Investigation Note:** {row['hidden_note']}
"""

# =========================================================
# Data generation
# =========================================================
def generate_data(n=24, seed=42):
    rng = np.random.default_rng(seed)
    subgroup = rng.binomial(1, 0.35, n) # 35% are "messy" cases

    data = []
    
    # Note pools (keeping your original lists internally...)
    pos_notes = ["Multiple failed CVV attempts", "Card reported stolen", "Shipping address changed recently", "Device linked to prior fraud"]
    neg_notes = ["Customer confirmed travel", "Trusted device", "Verified mobile push approval", "Matches recurring pattern"]
    neu_notes = ["No additional signals", "Metadata incomplete", "Limited account history"]

    for i in range(n):
        # Features
        amt, pf, ov = float(rng.uniform(20, 5000)), int(rng.integers(0, 6)), int(rng.binomial(1, 0.2))
        age, acc_age = int(rng.integers(18, 80)), int(rng.integers(5, 3650))
        
        # Determine "Human-Only" Signal
        note_type = rng.choice(["pos", "neg", "neu"], p=[0.25, 0.25, 0.50])
        note = rng.choice(pos_notes if note_type=="pos" else neg_notes if note_type=="neg" else neu_notes)
        signal = 1 if note_type=="pos" else -1 if note_type=="neg" else 0

        # Latent math: High weights + Negative bias = Decisive tails
        base_risk = (0.0005 * amt) + (1.6 * pf) + (2.2 * ov) - (0.0009 * acc_age) - 3.8
        
        # TWEAK: Subgroup noise makes 'uncertain' cases actually uncertain
        # If in subgroup, noise is higher, forcing scores toward the 0.5 middle
        model_noise = 0.65 if subgroup[i] == 1 else 0.15
        true_noise = 0.45 

        # Model only sees a fraction of the 'hidden' signal
        latent_model = base_risk + (0.4 * signal) + rng.normal(0, model_noise)
        # Truth depends heavily on the 'hidden' signal (what the human sees)
        latent_true = base_risk + (1.8 * signal) + rng.normal(0, true_noise)

        data.append({
            "case_id": i + 1,
            "model_score": round(float(sigmoid(latent_model)), 3),
            "true_label": int(rng.binomial(1, sigmoid(latent_true))),
            "amount": amt, "age": age, "prior_flags": pf, "account_age_days": acc_age,
            "overseas": ov, "hidden_note": note, "hidden_signal": signal
        })

    return pd.DataFrame(data)

# =========================================================
# Sidebar & Logic
# =========================================================
with st.sidebar:
    st.header("Simulation Settings")
    n_cases = st.slider("Transactions", 10, 60, 24)
    seed = st.number_input("Random Seed", 1, 9999, 42)
    
    st.markdown("---")
    low_thr = st.slider("Auto-Approve Below", 0.0, 0.45, 0.20)
    high_thr = st.slider("Auto-Block Above", 0.55, 1.0, 0.80)
    
    st.markdown("---")
    fn_cost = st.slider("Missed Fraud Cost (FN)", 1, 20, 10)
    review_cost = st.slider("Human Labor Cost", 0, 5, 1)
    
    if st.button("Generate New Data"):
        st.session_state.df = generate_data(n_cases, int(seed))

if "df" not in st.session_state:
    st.session_state.df = generate_data(n_cases, 42)

df = st.session_state.df.copy()

# Scoring Logic
df["needs_review"] = ((df["model_score"] > low_thr) & (df["model_score"] < high_thr)).astype(int)
df["model_only_pred"] = (df["model_score"] >= 0.5).astype(int)

# Distribution Chart (for integrity)
with st.sidebar:
    st.markdown("### Risk Score Distribution")
    counts, bins = np.histogram(df['model_score'], bins=10, range=(0,1))
    st.bar_chart(pd.DataFrame(counts, index=[f"{round(b,1)}" for b in bins[:-1]]))

# =========================================================
# Main UI
# =========================================================
st.title("Human-in-the-Loop: Fraud Decisioning")
st.markdown("Evaluating where automation ends and human judgment begins.")

col_a, col_b = st.columns([2, 1])
with col_a:
    st.subheader("Transaction Queue")
    display_df = df[["case_id", "amount", "model_score", "needs_review"]].copy()
    display_df["needs_review"] = display_df["needs_review"].map({1: "⚠️ Review", 0: "✅ Auto"})
    st.dataframe(display_df, use_container_width=True, hide_index=True)

with col_b:
    st.info(f"**Queue Stats:**\n\nTotal: {len(df)}\n\nAuto-Decided: {len(df) - df.needs_review.sum()}\n\nManual Review: {df.needs_review.sum()}")

st.divider()

# Human Review Section
st.subheader("Step 1: Manual Investigation")
review_cases = df[df["needs_review"] == 1]
human_decisions = {}

if len(review_cases) > 0:
    for _, row in review_cases.iterrows():
        with st.expander(f"Investigate Case {int(row['case_id'])}"):
            st.markdown(format_case_summary(row))
            choice = st.radio("Decision:", ["Allow", "Block"], key=f"d_{row.case_id}", horizontal=True)
            human_decisions[int(row["case_id"])] = 1 if choice == "Block" else 0
else:
    st.success("All transactions were handled automatically based on current thresholds.")

# Final Prediction Calculation
df["hitl_pred"] = [human_decisions.get(int(r.case_id), 1 if r.model_score >= high_thr else 0) for _, r in df.iterrows()]

# Results
if st.button("Calculate Final Performance"):
    st.divider()
    m_metrics = evaluate(df["model_only_pred"], df["true_label"])
    h_metrics = evaluate(df["hitl_pred"], df["true_label"])
    
    m_cost = expected_cost(df["model_only_pred"], df["true_label"], fn_cost=fn_cost)
    h_cost = expected_cost(df["hitl_pred"], df["true_label"], df["needs_review"], fn_cost=fn_cost, review_cost=review_cost)

    res_df = pd.DataFrame([
        {"System": "100% Automated", **m_metrics, "Cost": m_cost},
        {"System": "Human-in-the-Loop", **h_metrics, "Cost": h_cost}
    ])
    
    st.subheader("Step 2: Results")
    st.table(res_df)
    
    if h_cost < m_cost:
        st.success(f"Success! The HITL system saved ${m_cost - h_cost} compared to full automation.")
    else:
        st.warning("The human review cost outweighed the accuracy gains. Consider widening the thresholds.")

    with st.expander("View Case-by-Case Truth Labels"):
        st.write(df[["case_id", "model_score", "hitl_pred", "true_label"]].rename(columns={"true_label": "Actual Fraud? (1=Yes)"}))