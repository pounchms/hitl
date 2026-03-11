import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Human-in-the-Loop Decision Demo", layout="wide")

# =========================================================
# Data generation
# =========================================================
def generate_data(n=24, seed=42):
    rng = np.random.default_rng(seed)

    subgroup = rng.binomial(1, 0.30, n)
    true_label = rng.binomial(1, 0.25, n)

    scores = []
    ages = []
    prior_flags = []
    amount = []
    account_age_days = []
    overseas = []
    clues = []

    # Hidden Clue Logic: 
    # These are clues the human sees that are highly predictive, 
    # but the 'model_score' doesn't account for them.
    risky_clues = [
        "Shipping address doesn't match billing (High Risk)",
        "Device ID associated with previous fraud (High Risk)",
        "Rapid-fire clicks on checkout page (High Risk)",
        "Email address created 2 hours ago (High Risk)"
    ]
    safe_clues = [
        "Customer is a 'Verified Premium' member (Safe)",
        "Social media profile link is verified (Safe)",
        "Customer contacted support to confirm purchase (Safe)",
        "Long-standing history with no previous issues (Safe)"
    ]

    for i in range(n):
        ages.append(int(rng.integers(18, 80)))
        prior_flags.append(int(rng.integers(0, 6)))
        amount.append(round(float(rng.uniform(20, 5000)), 2))
        account_age_days.append(int(rng.integers(5, 3650)))
        overseas.append(int(rng.binomial(1, 0.20)))

        # Assign Clues (Human Advantage)
        # 80% chance the clue correctly identifies the truth
        if rng.random() < 0.80:
            clue = rng.choice(risky_clues) if true_label[i] == 1 else rng.choice(safe_clues)
        else:
            clue = rng.choice(safe_clues) if true_label[i] == 1 else rng.choice(risky_clues)
        clues.append(clue)

        # Model Score Logic (Model is blind to the 'clue' text)
        if subgroup[i] == 0:
            s = rng.beta(8, 3) if true_label[i] == 1 else rng.beta(2, 8)
        else:
            s = rng.beta(5, 4) if true_label[i] == 1 else rng.beta(4, 5)
        scores.append(float(s))

    df = pd.DataFrame({
        "case_id": np.arange(1, n + 1),
        "subgroup": subgroup,
        "true_label": true_label,
        "model_score": np.round(scores, 3),
        "age": ages,
        "prior_flags": prior_flags,
        "amount": amount,
        "account_age_days": account_age_days,
        "overseas": overseas,
        "human_clue": clues
    })
    return df

# ... [Evaluation helpers and expected_cost remain the same] ...

def evaluate(preds, truth):
    preds = np.array(preds)
    truth = np.array(truth)
    tp = int(np.sum((preds == 1) & (truth == 1)))
    tn = int(np.sum((preds == 0) & (truth == 0)))
    fp = int(np.sum((preds == 1) & (truth == 0)))
    fn = int(np.sum((preds == 0) & (truth == 1)))
    accuracy = np.mean(preds == truth)
    return {
        "Accuracy": round(float(accuracy), 3),
        "False Positives": fp,
        "False Negatives": fn
    }

def expected_cost(preds, truth, review_mask=None, fn_cost=10, fp_cost=2, review_cost=1):
    preds = np.array(preds)
    truth = np.array(truth)
    fp = np.sum((preds == 1) & (truth == 0))
    fn = np.sum((preds == 0) & (truth == 1))
    reviews = int(np.sum(review_mask)) if review_mask is not None else 0
    return int(fn * fn_cost + fp * fp_cost + reviews * review_cost)

def format_case_summary(row):
    overseas_text = "Yes" if row["overseas"] == 1 else "No"
    return f"""
**Case {int(row['case_id'])}**
- Transaction amount: **${row['amount']:,.2f}**
- Customer age: **{int(row['age'])}**
- Model-estimated risk score: **{row['model_score']:.3f}**
- **Internal Investigator Note (Hidden from Model):** > 🔎 {row['human_clue']}
"""

# =========================================================
# App Logic (Streamlit UI)
# =========================================================

st.title("Human-in-the-Loop: The 'Information Advantage' Demo")
st.markdown("Can you beat the model now that you have access to **Investigator Notes** that the model cannot see?")

# Sidebar
with st.sidebar:
    st.header("Simulation Controls")
    n_cases = st.slider("Number of simulated transactions", 10, 60, 24)
    seed = st.number_input("Random seed", 1, 9999, 42)
    st.markdown("---")
    low_thr = st.slider("Auto-approve below", 0.0, 0.49, 0.35, 0.01)
    high_thr = st.slider("Auto-block above", 0.51, 1.0, 0.65, 0.01)
    fn_cost = st.slider("Cost of False Negative", 1, 50, 20)
    fp_cost = st.slider("Cost of False Positive", 1, 20, 10)
    review_cost = st.slider("Cost of human review", 0, 5, 2)
    regenerate = st.button("Generate new cases")

if "df" not in st.session_state or regenerate:
    st.session_state.df = generate_data(n=n_cases, seed=int(seed))

df = st.session_state.df.copy()
df["model_only_pred"] = (df["model_score"] >= 0.5).astype(int)
df["needs_review"] = ((df["model_score"] > low_thr) & (df["model_score"] < high_thr)).astype(int)

# Human review section
st.subheader("Step 1: Review the uncertain cases")
review_cases = df[df["needs_review"] == 1].copy()
human_decisions = {}

if len(review_cases) == 0:
    st.info("No cases in the 'Uncertain' range. Adjust sliders to send more to review.")
else:
    for _, row in review_cases.iterrows():
        with st.expander(f"Review Case {int(row['case_id'])} (Score: {row['model_score']})"):
            st.markdown(format_case_summary(row))
            decision = st.radio(f"Decision for {int(row['case_id'])}", ["Allow", "Block"], key=f"r_{int(row['case_id'])}")
            human_decisions[int(row["case_id"])] = 0 if decision == "Allow" else 1

# Calculate Results
hitl_preds = [human_decisions.get(int(r["case_id"]), (1 if r["model_score"] >= high_thr else 0)) for _, r in df.iterrows()]
df["hitl_pred"] = hitl_preds

# Metrics Display
model_metrics = evaluate(df["model_only_pred"], df["true_label"])
hitl_metrics = evaluate(df["hitl_pred"], df["true_label"])

model_cost = expected_cost(df["model_only_pred"], df["true_label"], None, fn_cost, fp_cost, review_cost)
hitl_cost = expected_cost(df["hitl_pred"], df["true_label"], df["needs_review"], fn_cost, fp_cost, review_cost)

st.subheader("Step 2: Compare Results")
c1, c2 = st.columns(2)
c1.metric("Model-Only Cost", model_cost)
c2.metric("Human-in-the-Loop Cost", hitl_cost, delta=f"{hitl_cost - model_cost} vs Model", delta_color="inverse")

if st.checkbox("Show Decision Table"):
    st.dataframe(df[["case_id", "model_score", "human_clue", "model_only_pred", "hitl_pred", "true_label"]])