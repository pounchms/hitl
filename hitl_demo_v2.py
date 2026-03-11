import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Human-in-the-Loop Decision Demo", layout="wide")


# =========================================================
# Helpers
# =========================================================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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

**Additional human-only context**
- {row['hidden_note']}
"""


# =========================================================
# Data generation
# =========================================================
def generate_data(n=24, seed=42):
    rng = np.random.default_rng(seed)

    subgroup = rng.binomial(1, 0.30, n)  # harder subgroup where model is less reliable

    ages = []
    prior_flags = []
    amount = []
    account_age_days = []
    overseas = []
    hidden_note = []
    hidden_signal = []
    true_label = []
    model_score = []

    # Positive notes = stronger evidence of fraud / risk
    positive_notes = [
        "System Investigation Note: Multiple failed CVV attempts in logs",
        "System Investigation Note: Card reported stolen earlier today",
        "System Investigation Note: Shipping address changed 15 minutes ago",
        "System Investigation Note: Device fingerprint linked to prior fraud cases",
        "System Investigation Note: Billing ZIP mismatch detected during verification",
        "System Investigation Note: Several failed password attempts preceded this purchase",
        "System Investigation Note: Transaction originated from an IP address linked to prior fraud alerts",
        "System Investigation Note: Account email was changed within the last hour",
        "System Investigation Note: Phone number on file was updated immediately before checkout",
        "System Investigation Note: Card security checks failed twice before final authorization",
        "System Investigation Note: Unusual device-browser combination compared with account history",
        "System Investigation Note: Merchant category matches recent fraud pattern watchlist",
        "System Investigation Note: Velocity checks show multiple declined attempts before approval",
        "System Investigation Note: Shipping and billing names do not match account records",
        "System Investigation Note: Login occurred from a new country shortly before purchase",
        "System Investigation Note: Two-factor authentication was bypassed after repeated retries",
        "System Investigation Note: Purchase followed a password reset from an unfamiliar device",
        "System Investigation Note: Account recovery was triggered earlier today",
        "System Investigation Note: Card was used at multiple merchants within a very short window",
        "System Investigation Note: Device ID overlaps with accounts previously closed for fraud"
    ]

    # Negative notes = evidence supporting legitimacy / safety
    negative_notes = [
        "System Investigation Note: Customer confirmed travel through mobile app",
        "System Investigation Note: Purchase made from previously trusted device",
        "System Investigation Note: Customer responded YES to verification text",
        "System Investigation Note: Merchant is on recurring subscription whitelist",
        "System Investigation Note: Transaction matches the customer's normal travel pattern",
        "System Investigation Note: Same merchant used successfully by this customer in prior months",
        "System Investigation Note: Device has been associated with verified account activity for over a year",
        "System Investigation Note: Customer recently notified bank of planned international travel",
        "System Investigation Note: Purchase matches recurring monthly spending pattern",
        "System Investigation Note: Shipping address matches long-standing saved address",
        "System Investigation Note: Verified mobile push approval received before checkout",
        "System Investigation Note: Merchant is on customer favorites list",
        "System Investigation Note: Customer support note confirms this purchase category is expected",
        "System Investigation Note: Geolocation is consistent with customer's recent confirmed activity",
        "System Investigation Note: Transaction made from customer’s usual home network",
        "System Investigation Note: Purchase amount aligns with normal paycheck-cycle spending",
        "System Investigation Note: Customer recently added this merchant to approved vendors",
        "System Investigation Note: Device and browser match most recent successful login session",
        "System Investigation Note: Travel itinerary on file supports current transaction location",
        "System Investigation Note: Previous manual review cleared a nearly identical purchase"
    ]

    # Neutral notes = ambiguous / incomplete / mixed signals
    neutral_notes = [
        "System Investigation Note: No additional signals found",
        "System Investigation Note: Manual review history unavailable",
        "System Investigation Note: Device metadata incomplete",
        "System Investigation Note: No recent account changes detected",
        "System Investigation Note: Merchant history is limited for this account",
        "System Investigation Note: Customer profile has no recent analyst comments",
        "System Investigation Note: Fraud monitoring logs are partially unavailable",
        "System Investigation Note: Device risk service returned inconclusive result",
        "System Investigation Note: Verification event timing could not be confirmed",
        "System Investigation Note: Purchase context appears mixed across available systems",
        "System Investigation Note: Network reputation signal unavailable at decision time",
        "System Investigation Note: Prior transaction history is sparse",
        "System Investigation Note: Merchant risk feed returned no actionable flags",
        "System Investigation Note: Customer response history is unavailable",
        "System Investigation Note: Session logs are incomplete for this transaction",
        "System Investigation Note: Address consistency check returned an inconclusive result",
        "System Investigation Note: Historical baseline could not be computed reliably",
        "System Investigation Note: Limited cross-device history available",
        "System Investigation Note: Authentication logs show no clear anomaly",
        "System Investigation Note: Internal monitoring systems returned mixed signals"
    ]

    for i in range(n):
        # Observable features
        a = int(rng.integers(18, 80))
        pf = int(rng.integers(0, 6))
        amt = round(float(rng.uniform(20, 5000)), 2)
        acc_age = int(rng.integers(5, 3650))
        ov = int(rng.binomial(1, 0.20))

        ages.append(a)
        prior_flags.append(pf)
        amount.append(amt)
        account_age_days.append(acc_age)
        overseas.append(ov)

        # Hidden human-only note type
        note_type = rng.choice(["positive", "negative", "neutral"], p=[0.24, 0.24, 0.52])

        if note_type == "positive":
            note = rng.choice(positive_notes)
            signal = 1
        elif note_type == "negative":
            note = rng.choice(negative_notes)
            signal = -1
        else:
            note = rng.choice(neutral_notes)
            signal = 0

        hidden_note.append(note)
        hidden_signal.append(signal)

        # Observable risk
        observable_risk = (
            0.00010 * amt
            + 0.22 * pf
            + 0.75 * ov
            - 0.00045 * acc_age
            + 0.002 * max(0, a - 65)
        )

        # Harder subgroup has more noise
        subgroup_noise_true = 0.55 if subgroup[i] == 1 else 0.30
        subgroup_noise_model = 0.45 if subgroup[i] == 1 else 0.28

        # Truth depends strongly on hidden signal
        latent_true = observable_risk + 1.25 * signal + rng.normal(0, subgroup_noise_true)
        prob_true = sigmoid(latent_true)
        y = int(rng.binomial(1, prob_true))
        true_label.append(y)

        # Model only weakly captures the hidden signal
        latent_model = observable_risk + 0.20 * signal + rng.normal(0, subgroup_noise_model)
        score = float(sigmoid(latent_model))
        model_score.append(round(score, 3))

    df = pd.DataFrame({
        "case_id": np.arange(1, n + 1),
        "subgroup": subgroup,
        "true_label": true_label,
        "model_score": model_score,
        "age": ages,
        "prior_flags": prior_flags,
        "amount": amount,
        "account_age_days": account_age_days,
        "overseas": overseas,
        "hidden_note": hidden_note,
        "hidden_signal": hidden_signal
    })

    return df


# =========================================================
# Page header
# =========================================================
st.title("Designing a Human-in-the-Loop Decision System")
st.subheader("When should a model decide on its own, and when should a human step in?")

st.markdown("""
This interactive demo simulates a **fraud screening system**.

A model reviews transactions and assigns each one a **risk score** between 0 and 1:

- A score near **0** means the transaction looks low risk
- A score near **1** means the transaction looks high risk

The real design question is not just whether the model is accurate.

> **Should the model make every decision automatically, or should uncertain cases be sent to a human reviewer?**

In this demo, you will compare two systems:

- **Model-only system:** the model decides every case by itself
- **Human-in-the-loop system:** the model handles clear cases, and **you review the uncertain ones**
""")

st.markdown("""
### How to use this demo

1. Click **Generate new simulated cases**.
2. Review the transactions marked **Needs Human Review**.
3. Use the investigation notes to decide whether to allow or block each case.
4. Compare the results between:
   - **Model decides everything**
   - **Human-in-the-loop system**
5. Reveal the true outcomes to see which approach performed better.
""")

st.info("""
In this demo, the human reviewer sees an additional investigation note that the model does not fully incorporate into its prediction.

This simulates real-world systems where human analysts may have access to contextual information that is difficult to encode directly in machine learning features.
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

The app also simulates a hidden harder-to-classify subgroup where the model is less reliable.

Most importantly, the human reviewer gets access to an extra **investigation note** that the model does not fully use. This means the human has a real chance to improve decisions on borderline cases.
""")

with st.expander("Why include a human reviewer at all?"):
    st.markdown("""
Human reviewers can add value in situations where a model is uncertain or missing context.

In this simulation, the reviewer receives an **additional investigation note** that the model does not fully incorporate into its prediction.

This represents information that may exist in real systems but is difficult to encode directly in model features.

Examples include:
- recent security alerts
- manual investigation logs
- customer confirmations
- device fingerprint intelligence

Human-in-the-loop systems work best when humans have **contextual information or judgment that the model does not fully capture**.
""")

with st.expander("What should you look for in the results?"):
    st.markdown("""
A human reviewer does **not** need to beat the model on every case.

Instead, the goal is for human review to add enough value on the **right cases** to justify its cost.

That means asking:

- Did human review reduce costly false negatives?
- Did it reduce false positives?
- How many cases needed review?
- Was the improvement large enough to offset review overhead?
""")


# =========================================================
# Sidebar controls
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

# Friendly display table (without hidden note)
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

These reviewed cases include an additional **human-only investigation note**.
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

st.markdown("""
### Why review cost matters
Human review is not free. It adds time, labor cost, and inconsistency.
A human-in-the-loop system is only worthwhile when the improvement in decisions
is large enough to offset that overhead.
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
# Final takeaway
# =========================================================
st.subheader("Main lesson")
st.markdown("""
A model can look strong on average and still be risky if it decides **every case by itself**.

The key question is not only:

> **How accurate is the model?**

It is also:

> **Which cases are safe to automate, and which cases should be reviewed by a human?**
""")

st.markdown("""
Two important lessons from this simulation are:

**1. Human oversight only adds value when the human has meaningful context or judgment beyond the model.**

**2. Human review introduces cost and delay, so the benefit must outweigh that overhead.**

Effective human-AI systems therefore focus on **routing the right decisions to the right decision maker**.
""")