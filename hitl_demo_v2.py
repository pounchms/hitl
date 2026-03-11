import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Human-in-the-Loop Decision Demo", layout="wide")

# =========================================================
# Data generation (MODIFIED to include Hidden Clues)
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

    # Hidden Clues: High-signal data points the model doesn't "see"
    risky_clues = [
        "Disposable email provider detected",
        "Multiple failed CVV attempts in logs",
        "Shipping address is a known freight forwarder",
        "Hardware ID linked to previous chargeback"
    ]
    safe_clues = [
        "Customer is a verified repeat buyer",
        "Billing address matches IP geolocation",
        "Linked to a 5-year-old social media profile",
        "Authorized by phone via 2FA"
    ]

    for i in range(n):
        ages.append(int(rng.integers(18, 80)))
        prior_flags.append(int(rng.integers(0, 6)))
        amount.append(round(float(rng.uniform(20, 5000)), 2))
        account_age_days.append(int(rng.integers(5, 3650)))
        overseas.append(int(rng.binomial(1, 0.20)))

        # Assign Clue: 80% chance the clue reflects the TRUTH
        if rng.random() < 0.80:
            clue = rng.choice(risky_clues) if true_label[i] == 1 else rng.choice(safe_clues)
        else:
            clue = rng.choice(safe_clues) if true_label[i] == 1 else rng.choice(risky_clues)
        clues.append(clue)

        # Model score remains the same (blind to the clue)
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
        "human_clue": clues # The "Hidden Data"
    })

    return df


# =========================================================
# Evaluation helpers (No changes here)
# =========================================================
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

# MODIFIED to show the hidden clue to the human reviewer
def format_case_summary(row):
    overseas_text = "Yes" if row["overseas"] == 1 else "No"
    return f"""
**Case {int(row['case_id'])}**

- Transaction amount: **${row['amount']:,.2f}**
- Customer age: **{int(row['age'])}**
- Model-estimated risk score: **{row['model_score']:.3f}**
- **System Investigation Note:** *{row['human_clue']}* """

# =========================================================
# Main App Logic (Rest remains consistent with your code)
# =========================================================

st.title("Designing a Human-in-the-Loop Decision System")

# Sidebar and session state logic as per your original script
with st.sidebar:
    st.header("Simulation Controls")
    n_cases = st.slider("Number of simulated transactions", 10, 60, 24)
    seed = st.number_input("Random seed", min_value=1, max_value=9999, value=42)
    st.markdown("---")
    st.markdown("### Human-in-the-loop routing rules")
    low_thr = st.slider("Auto-approve below", 0.0, 0.49, 0.20, 0.01)
    high_thr = st.slider("Auto-block above", 0.51, 1.0, 0.80, 0.01)
    st.markdown("---")
    st.markdown("### Cost assumptions")
    fn_cost = st.slider("Cost of False Negative", 1, 20, 10)
    fp_cost = st.slider("Cost of False Positive", 1, 10, 2)
    review_cost = st.slider("Cost of human review", 0, 5, 1)
    regenerate = st.button("Generate new simulated cases")

if "df" not in st.session_state or regenerate:
    st.session_state.df = generate_data(n=n_cases, seed=int(seed))

df = st.session_state.df.copy()
df["model_only_pred"] = (df["model_score"] >= 0.5).astype(int)
df["needs_review"] = ((df["model_score"] > low_thr) & (df["model_score"] < high_thr)).astype(int)

# Use map to fill auto-decisions
def auto_hitl_decision(score):
    if score <= low_thr: return 0
    elif score >= high_thr: return 1
    return None
df["hitl_auto_pred"] = df["model_score"].apply(auto_hitl_decision)

st.subheader("Step 1: Review the uncertain cases")
review_cases = df[df["needs_review"] == 1].copy()
human_decisions = {}

if len(review_cases) == 0:
    st.info("No cases need review. Try narrowing the auto-approve/block ranges.")
else:
    for _, row in review_cases.iterrows():
        with st.expander(f"Review Case {int(row['case_id'])}"):
            st.markdown(format_case_summary(row))
            decision = st.radio(f"Action for Case {int(row['case_id'])}", ["Allow", "Block"], key=f"rev_{int(row['case_id'])}")
            human_decisions[int(row["case_id"])] = 0 if decision == "Allow" else 1

# Merge human decisions back
hitl_preds = []
for _, row in df.iterrows():
    if row["needs_review"] == 0:
        hitl_preds.append(int(row["hitl_auto_pred"]))
    else:
        hitl_preds.append(human_decisions.get(int(row["case_id"]), 0))
df["hitl_pred"] = hitl_preds

# Results Section
model_metrics = evaluate(df["model_only_pred"], df["true_label"])
hitl_metrics = evaluate(df["hitl_pred"], df["true_label"])
model_cost = expected_cost(df["model_only_pred"], df["true_label"], np.zeros(len(df)), fn_cost, fp_cost, review_cost)
hitl_cost = expected_cost(df["hitl_pred"], df["true_label"], df["needs_review"], fn_cost, fp_cost, review_cost)

st.subheader("Step 2: Compare the two decision systems")
res_df = pd.DataFrame([
    {"System": "Automated", **model_metrics, "Cost": model_cost},
    {"System": "Human-in-the-Loop", **hitl_metrics, "Cost": hitl_cost}
])
st.table(res_df)

if st.checkbox("Step 3: Reveal Outcomes"):
    st.dataframe(df[["case_id", "model_score", "human_clue", "hitl_pred", "true_label"]])