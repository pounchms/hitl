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

        # UPDATED: Increased weights and added negative bias to push scores toward 0/1
        observable_risk = (
            0.00050 * amt           
            + 1.50 * pf             
            + 2.00 * ov             
            - 0.0008 * acc_age      
            + 0.005 * max(0, a - 65)
            - 3.5                   # Strong negative bias (default = safe)
        )

        subgroup_noise_true = 0.55 if subgroup[i] == 1 else 0.30
        
        # UPDATED: Lower noise for model confidence
        subgroup_noise_model = 0.25 if subgroup[i] == 1 else 0.12

        latent_true = observable_risk + 1.25 * signal + rng.normal(0, subgroup_noise_true)
        prob_true = sigmoid(latent_true)
        y = int(rng.binomial(1, prob_true))
        true_label.append(y)

        # Model score calculation
        latent_model = observable_risk + 0.50 * signal + rng.normal(0, subgroup_noise_model)
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
# Page layout & UI
# =========================================================
st.title("Designing a Human-in-the-Loop Decision System")
st.subheader("When should a model decide on its own, and when should a human step in?")

# ... (Markdown and expander sections remain identical to your original code)

# =========================================================
# Sidebar controls
# =========================================================
with st.sidebar:
    st.header("Simulation Controls")
    n_cases = st.slider("Number of simulated transactions", 10, 100, 24)
    seed = st.number_input("Random seed", min_value=1, max_value=9999, value=42)

    st.markdown("---")
    st.markdown("### Human-in-the-loop routing rules")
    low_thr = st.slider("Auto-approve below", 0.0, 0.49, 0.20, 0.01)
    high_thr = st.slider("Auto-block above", 0.51, 1.0, 0.80, 0.01)

    st.markdown("---")
    st.markdown("### Cost assumptions")
    fn_cost = st.slider("Cost of missing a risky case (FN)", 1, 20, 10)
    fp_cost = st.slider("Cost of blocking a safe case (FP)", 1, 10, 2)
    review_cost = st.slider("Cost of human review", 0, 5, 1)

    regenerate = st.button("Generate new simulated cases")

if low_thr >= high_thr:
    st.error("The auto-approve threshold must be lower than the auto-block threshold.")
    st.stop()


# =========================================================
# Session state & Prediction Logic
# =========================================================
if "df" not in st.session_state or regenerate:
    st.session_state.df = generate_data(n=n_cases, seed=int(seed))

df = st.session_state.df.copy()

df["model_only_pred"] = (df["model_score"] >= 0.5).astype(int)
df["needs_review"] = ((df["model_score"] > low_thr) & (df["model_score"] < high_thr)).astype(int)

def auto_hitl_decision(score):
    if score <= low_thr: return 0
    elif score >= high_thr: return 1
    return None

df["hitl_auto_pred"] = df["model_score"].apply(auto_hitl_decision)

# Display Table
display_df = df[[
    "case_id", "amount", "age", "prior_flags", "account_age_days", "overseas", "model_score", "needs_review"
]].copy().rename(columns={
    "case_id": "Case ID", "amount": "Amount ($)", "age": "Customer Age", 
    "prior_flags": "Prior Flags", "account_age_days": "Account Age (days)", 
    "overseas": "International?", "model_score": "Model Risk Score", 
    "needs_review": "Needs Human Review?"
})
display_df["International?"] = display_df["International?"].map({0: "No", 1: "Yes"})
display_df["Needs Human Review?"] = display_df["Needs Human Review?"].map({0: "No", 1: "Yes"})

st.subheader("Simulated transactions")
st.dataframe(display_df, use_container_width=True)


# =========================================================
# Human review section
# =========================================================
st.subheader("Step 1: Review the uncertain cases")
review_cases = df[df["needs_review"] == 1].copy()
human_decisions = {}

if len(review_cases) == 0:
    st.info("The model is very confident! No cases currently need human review.")
else:
    for _, row in review_cases.iterrows():
        with st.expander(f"Review Case {int(row['case_id'])}"):
            st.markdown(format_case_summary(row))
            decision = st.radio(
                f"Action for Case {int(row['case_id'])}?",
                options=["Allow transaction", "Flag / block transaction"],
                key=f"review_{int(row['case_id'])}"
            )
            human_decisions[int(row["case_id"])] = 0 if decision == "Allow transaction" else 1

# Final Predictions
hitl_preds = [int(row["hitl_auto_pred"]) if row["needs_review"] == 0 
              else human_decisions.get(int(row["case_id"]), 0) for _, row in df.iterrows()]
df["hitl_pred"] = hitl_preds


# =========================================================
# Results & Comparison
# =========================================================
st.subheader("Step 2: Compare the two decision systems")

model_metrics = evaluate(df["model_only_pred"], df["true_label"])
hitl_metrics = evaluate(df["hitl_pred"], df["true_label"])

model_cost = expected_cost(df["model_only_pred"], df["true_label"], np.zeros(len(df)), fn_cost, fp_cost, review_cost)
hitl_cost = expected_cost(df["hitl_pred"], df["true_label"], df["needs_review"], fn_cost, fp_cost, review_cost)

results = pd.DataFrame([
    {"System": "Model-only", **model_metrics, "Review Rate": 0.0, "Exp. Cost": model_cost},
    {"System": "Human-in-the-loop", **hitl_metrics, "Review Rate": round(float(df["needs_review"].mean()), 3), "Exp. Cost": hitl_cost}
])
st.dataframe(results, use_container_width=True)

# Outcome Reveal Logic
st.subheader("Step 3: Reveal the true outcomes")
reveal = st.checkbox("Show truth labels")
if reveal:
    st.write(df[["case_id", "model_score", "model_only_pred", "hitl_pred", "true_label"]])