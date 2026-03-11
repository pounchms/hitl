import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Human-in-the-Loop Decision Demo", layout="wide")

# =========================================================
# Data generation
# =========================================================
def generate_data(n=24, seed=42):
    rng = np.random.default_rng(seed)

    # Simulated subgroup indicator:
    # 0 = standard cases
    # 1 = harder-to-classify cases where the model is less reliable
    subgroup = rng.binomial(1, 0.30, n)

    # True outcome:
    # 1 = truly high risk
    # 0 = truly low risk
    true_label = rng.binomial(1, 0.25, n)

    scores = []
    ages = []
    prior_flags = []
    amount = []
    account_age_days = []
    overseas = []

    for i in range(n):
        ages.append(int(rng.integers(18, 80)))
        prior_flags.append(int(rng.integers(0, 6)))
        amount.append(round(float(rng.uniform(20, 5000)), 2))
        account_age_days.append(int(rng.integers(5, 3650)))
        overseas.append(int(rng.binomial(1, 0.20)))

        if subgroup[i] == 0:
            if true_label[i] == 1:
                s = rng.beta(8, 3)
            else:
                s = rng.beta(2, 8)
        else:
            if true_label[i] == 1:
                s = rng.beta(5, 4)
            else:
                s = rng.beta(4, 5)

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
        "overseas": overseas
    })

    return df


# =========================================================
# Evaluation helpers
# =========================================================
def evaluate(preds, truth):
    preds = np.array(preds)
    truth = np.array(truth)

    tp = int(np.sum((preds == 1) & (truth == 1)))
    tn = int(np.sum((preds == 0) & (truth == 0)))
    fp = int(np.sum((preds == 1) & (truth == 0)))
    fn = int(np.sum((preds == 0) & (truth == 1)))

    accuracy = np.mean(preds == truth)
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "Accuracy": round(float(accuracy), 3),
        "Sensitivity": round(float(sensitivity), 3),
        "Specificity": round(float(specificity), 3),
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
- Number of prior account flags: **{int(row['prior_flags'])}**
- Account age: **{int(row['account_age_days'])} days**
- International transaction: **{overseas_text}**
- Model-estimated risk score: **{row['model_score']:.3f}**
"""


# =========================================================
# App title and introduction
# =========================================================
st.title("Designing a Human-in-the-Loop Decision System")
st.subheader("When should a model decide on its own, and when should a human step in?")

st.markdown("""
This interactive demo simulates a **fraud screening system**.

A model reviews transactions and assigns each one a **risk score** between 0 and 1:

- A score near **0** means the transaction looks low risk
- A score near **1** means the transaction looks high risk

The question is not just whether the model is accurate.

The real design question is:

> **Should the model make every decision automatically, or should uncertain cases be sent to a human reviewer?**

In this demo, you will compare two systems:

- **Model-only system:** the model decides every case by itself
- **Human-in-the-loop system:** the model handles clear cases, and **you review the uncertain ones**
""")

with st.expander("What are we simulating?"):
    st.markdown("""
We are using **simulated data**, not real customer data.

Each case represents a fictional transaction with features such as:

- transaction amount
- customer age
- number of prior account flags
- account age
- whether the transaction is international

The app also simulates a hidden “harder” subgroup where the model is less reliable.
That is important because a model can look strong **on average** while still making worse decisions in certain kinds of cases.

This is one reason human oversight can matter.
""")

with st.expander("How should I interpret the risk score?"):
    st.markdown("""
The risk score is the model's estimate of how likely a case is to be high risk.

For example:

- **0.10** = the model thinks the case is probably safe
- **0.52** = the model is unsure
- **0.93** = the model thinks the case is probably risky

A common mistake in automated systems is to treat these scores as if they were certain facts.
In practice, borderline cases often deserve human review.
""")


# =========================================================
# Sidebar
# =========================================================
with st.sidebar:
    st.header("Simulation Controls")
    n_cases = st.slider("Number of simulated transactions", 10, 60, 24)
    seed = st.number_input("Random seed", min_value=1, max_value=9999, value=42)

    st.markdown("---")
    st.markdown("### Human-in-the-loop routing rules")
    low_thr = st.slider("Auto-approve below", 0.0, 0.49, 0.20, 0.01)
    high_thr = st.slider("Auto-block above", 0.51, 1.0, 0.80, 0.01)

    st.caption("Scores in the middle are sent to human review.")

    st.markdown("---")
    st.markdown("### Cost assumptions")
    fn_cost = st.slider("Cost of missing a risky case (false negative)", 1, 20, 10)
    fp_cost = st.slider("Cost of blocking a safe case (false positive)", 1, 10, 2)
    review_cost = st.slider("Cost of human review", 0, 5, 1)

    regenerate = st.button("Generate new simulated cases")

if low_thr >= high_thr:
    st.error("The auto-approve threshold must be lower than the auto-block threshold.")
    st.stop()


# =========================================================
# Session state
# =========================================================
if "df" not in st.session_state or regenerate:
    st.session_state.df = generate_data(n=n_cases, seed=int(seed))

df = st.session_state.df.copy()

# Model-only baseline
df["model_only_pred"] = (df["model_score"] >= 0.5).astype(int)

# HITL routing
df["needs_review"] = ((df["model_score"] > low_thr) & (df["model_score"] < high_thr)).astype(int)

def auto_hitl_decision(score):
    if score <= low_thr:
        return 0
    elif score >= high_thr:
        return 1
    return None

df["hitl_auto_pred"] = df["model_score"].apply(auto_hitl_decision)

# Friendly display table
display_df = df[[
    "case_id", "amount", "age", "prior_flags", "account_age_days", "overseas", "model_score", "needs_review"
]].copy()

display_df = display_df.rename(columns={
    "case_id": "Case ID",
    "amount": "Amount ($)",
    "age": "Customer Age",
    "prior_flags": "Prior Flags",
    "account_age_days": "Account Age (days)",
    "overseas": "International?",
    "model_score": "Model Risk Score",
    "needs_review": "Needs Human Review?"
})

display_df["International?"] = display_df["International?"].map({0: "No", 1: "Yes"})
display_df["Needs Human Review?"] = display_df["Needs Human Review?"].map({0: "No", 1: "Yes"})

st.subheader("Simulated transactions")
st.markdown("""
These are the transactions the system is evaluating.

The model risk score is shown for every case.  
Cases in the uncertain middle range are routed to **human review**.
""")
st.dataframe(display_df, use_container_width=True)

# =========================================================
# Human review section
# =========================================================
st.subheader("Step 1: Review the uncertain cases")
st.markdown("""
Below are the cases the system is **not confident enough to decide automatically**.

For each one, choose:

- **Allow transaction** if you think it is likely safe
- **Flag / block transaction** if you think it is likely risky
""")

review_cases = df[df["needs_review"] == 1].copy()
human_decisions = {}

if len(review_cases) == 0:
    st.info("No cases need human review under the current threshold settings.")
else:
    for _, row in review_cases.iterrows():
        with st.expander(f"Review Case {int(row['case_id'])}"):
            st.markdown(format_case_summary(row))

            decision = st.radio(
                f"What should happen to Case {int(row['case_id'])}?",
                options=["Allow transaction", "Flag / block transaction"],
                key=f"review_{int(row['case_id'])}"
            )

            human_decisions[int(row["case_id"])] = 0 if decision == "Allow transaction" else 1

# =========================================================
# Final HITL predictions
# =========================================================
hitl_preds = []

for _, row in df.iterrows():
    if row["needs_review"] == 0:
        hitl_preds.append(int(row["hitl_auto_pred"]))
    else:
        case_id = int(row["case_id"])
        hitl_preds.append(human_decisions.get(case_id, 0))

df["hitl_pred"] = hitl_preds

# =========================================================
# Results
# =========================================================
st.subheader("Step 2: Compare the two decision systems")

model_metrics = evaluate(df["model_only_pred"], df["true_label"])
hitl_metrics = evaluate(df["hitl_pred"], df["true_label"])

model_cost = expected_cost(
    df["model_only_pred"],
    df["true_label"],
    review_mask=np.zeros(len(df), dtype=int),
    fn_cost=fn_cost,
    fp_cost=fp_cost,
    review_cost=review_cost
)

hitl_cost = expected_cost(
    df["hitl_pred"],
    df["true_label"],
    review_mask=df["needs_review"],
    fn_cost=fn_cost,
    fp_cost=fp_cost,
    review_cost=review_cost
)

results = pd.DataFrame([
    {
        "System": "Model decides everything",
        **model_metrics,
        "Human Review Rate": 0.000,
        "Expected Cost": model_cost
    },
    {
        "System": "Human-in-the-loop",
        **hitl_metrics,
        "Human Review Rate": round(float(df["needs_review"].mean()), 3),
        "Expected Cost": hitl_cost
    }
])

st.dataframe(results, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Before: Full automation")
    st.markdown(f"""
In this version, the model decides **every case** using a 0.50 cutoff.

- False positives: **{model_metrics['False Positives']}**
- False negatives: **{model_metrics['False Negatives']}**
- Accuracy: **{model_metrics['Accuracy']}**
- Expected cost: **{model_cost}**
""")

with col2:
    st.markdown("### After: Human in the loop")
    st.markdown(f"""
In this version, the model handles the easy cases and **you review the uncertain ones**.

- Human review rate: **{round(float(df['needs_review'].mean()), 3)}**
- False positives: **{hitl_metrics['False Positives']}**
- False negatives: **{hitl_metrics['False Negatives']}**
- Accuracy: **{hitl_metrics['Accuracy']}**
- Expected cost: **{hitl_cost}**
""")

# =========================================================
# Reveal outcomes
# =========================================================
st.subheader("Step 3: Reveal the true outcomes")
st.markdown("""
You can now compare what the model would have done versus what happened when a human reviewed uncertain cases.
""")

reveal = st.checkbox("Show true outcomes")

case_results = df[[
    "case_id", "model_score", "needs_review", "model_only_pred", "hitl_pred", "true_label"
]].copy()

case_results = case_results.rename(columns={
    "case_id": "Case ID",
    "model_score": "Model Risk Score",
    "needs_review": "Sent to Human Review?",
    "model_only_pred": "Model-Only Decision",
    "hitl_pred": "Human-in-the-Loop Decision",
    "true_label": "True Outcome"
})

case_results["Sent to Human Review?"] = case_results["Sent to Human Review?"].map({0: "No", 1: "Yes"})
case_results["Model-Only Decision"] = case_results["Model-Only Decision"].map({0: "Allow", 1: "Block"})
case_results["Human-in-the-Loop Decision"] = case_results["Human-in-the-Loop Decision"].map({0: "Allow", 1: "Block"})
case_results["True Outcome"] = case_results["True Outcome"].map({0: "Actually safe", 1: "Actually risky"})

if reveal:
    st.dataframe(case_results, use_container_width=True)
else:
    st.dataframe(case_results.drop(columns=["True Outcome"]), use_container_width=True)

# =========================================================
# Teaching takeaway
# =========================================================
st.subheader("Main lesson")
st.markdown("""
A model can look strong on average and still be risky if it decides **every case by itself**.

The key question is not only:

> **How accurate is the model?**

It is also:

> **Which cases are safe to automate, and which cases should be reviewed by a human?**

That is the central idea behind designing a **human-in-the-loop decision system**.
""")

#Run this line in terminal to launch:
#streamlit run "C:\Users\Matt\OneDrive\VCU MDA\Spring2026\DAPT622_Statistics_II\Exploratory Analysis 2\hitl_demo.py"