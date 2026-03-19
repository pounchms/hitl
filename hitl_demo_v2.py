import streamlit as st
import numpy as np
import pandas as pd

# =========================================================
# Configuration & UI Styling
# =========================================================
st.set_page_config(page_title="HITL Decision Simulation", layout="wide")

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

# =========================================================
# Data Generation (Tuned for Presentation)
# =========================================================
def generate_data(n=24, seed=42):
    rng = np.random.default_rng(seed)
    subgroup = rng.binomial(1, 0.35, n) # 35% are "messy" cases for human review

    data = []
    
    # Text pools for investigation notes
    pos_notes = ["Multiple failed CVV attempts", "Card reported stolen", "Shipping address changed recently", "Device linked to prior fraud"]
    neg_notes = ["Customer confirmed travel", "Trusted device", "Verified mobile push approval", "Matches recurring pattern"]
    neu_notes = ["No additional signals", "Metadata incomplete", "Limited account history"]

    for i in range(n):
        # Features
        amt, pf, ov = float(rng.uniform(20, 5000)), int(rng.integers(0, 6)), int(rng.binomial(1, 0.2))
        age, acc_age = int(rng.integers(18, 80)), int(rng.integers(5, 3650))
        
        # Human-Only Signal
        note_type = rng.choice(["pos", "neg", "neu"], p=[0.25, 0.25, 0.50])
        note = rng.choice(pos_notes if note_type=="pos" else neg_notes if note_type=="neg" else neu_notes)
        signal = 1 if note_type=="pos" else -1 if note_type=="neg" else 0

        # Math logic: Negative bias (-3.8) + High weights = Decisive tails
        # We model the latent logit 'z' where P(y=1) = sigmoid(z)
        base_risk = (0.0005 * amt) + (1.6 * pf) + (2.2 * ov) - (0.0009 * acc_age) - 3.8
        
        # INCREASED NOISE for the subgroup pushes them toward the 0.5 center
        model_noise = 0.65 if subgroup[i] == 1 else 0.15
        true_noise = 0.45 

        # Model sees a weak version of the signal; Human sees the full note (stronger signal)
        latent_model = base_risk + (0.4 * signal) + rng.normal(0, model_noise)
        latent_true = base_risk + (1.8 * signal) + rng.normal(0, true_noise)

        data.append({
            "case_id": i + 1,
            "model_score": round(float(sigmoid(latent_model)), 3),
            "true_label": int(rng.binomial(1, sigmoid(latent_true))),
            "amount": amt, "age": age, "prior_flags": pf, "account_age_days": acc_age,
            "overseas": ov, "hidden_note": f"System Investigation Note: {note}", "hidden_signal": signal
        })

    return pd.DataFrame(data)

# =========================================================
# Sidebar Controls
# =========================================================
with st.sidebar:
    st.header("Simulation Controls")
    n_cases = st.slider("Number of simulated transactions", 10, 100, 24)
    seed = st.number_input("Random seed", min_value=1, max_value=9999, value=42)

    st.markdown("---")
    st.markdown("### Routing Rules")
    low_thr = st.slider("Auto-approve below", 0.0, 0.49, 0.20, 0.01)
    high_thr = st.slider("Auto-block above", 0.51, 1.0, 0.80, 0.01)
    st.caption("Scores in the middle are sent to human review.")

    st.markdown("---")
    st.markdown("### Cost Assumptions")
    fn_cost = st.slider("Cost of missing a risky case (FN)", 1, 20, 10)
    fp_cost = st.slider("Cost of blocking a safe case (FP)", 1, 10, 2)
    review_cost = st.slider("Cost of human review", 0, 5, 1)

    regenerate = st.button("Generate new simulated cases")

# =========================================================
# Main Page Header & Explanations
# =========================================================
st.title("Designing a Human-in-the-Loop Decision System")
st.subheader("When should a model decide on its own, and when should a human step in?")

st.markdown("""
This interactive demo simulates a **fraud screening system**. A model reviews transactions and assigns each one a **risk score** between 0 and 1.

> **The Design Challenge:** Should the model make every decision automatically, or should uncertain cases be sent to a human reviewer?

In this demo, you will compare:
1.  **Model-only system:** The model decides every case using a default 0.5 cutoff.
2.  **Human-in-the-loop (HITL):** The model handles clear cases; **you review the uncertain ones.**
""")

with st.expander("Why include a human reviewer at all?"):
    st.markdown("""
Human reviewers add value when a model is missing context. In this simulation, the reviewer sees an **additional investigation note** that the model does not fully incorporate.

Human-in-the-loop systems work best when humans have **contextual judgment** that is difficult to encode as machine learning features (e.g., nuanced security alerts or manual logs).
""")

with st.expander("What should you look for in the results?"):
    st.markdown("""
The goal isn't for the human to beat the model on every case. The goal is for human review to add enough value on **borderline cases** to justify its labor cost. 

Ask yourself:
- Did human review reduce costly false negatives?
- Was the improvement large enough to offset the human review overhead?
""")

# =========================================================
# Session State & Logic Processing
# =========================================================
if "df" not in st.session_state or regenerate:
    st.session_state.df = generate_data(n=n_cases, seed=int(seed))

df = st.session_state.df.copy()

# Integrity Check: Sidebar Histogram
with st.sidebar:
    st.markdown("### Model Score Distribution")
    counts, bins = np.histogram(df['model_score'], bins=10, range=(0,1))
    st.bar_chart(pd.DataFrame(counts, index=[f"{round(b,1)}" for b in bins[:-1]]))

# Routing Logic
df["needs_review"] = ((df["model_score"] > low_thr) & (df["model_score"] < high_thr)).astype(int)
df["model_only_pred"] = (df["model_score"] >= 0.5).astype(int)

# Friendly display table
display_df = df[["case_id", "amount", "age", "prior_flags", "account_age_days", "overseas", "model_score", "needs_review"]].copy()
display_df = display_df.rename(columns={"case_id": "ID", "model_score": "Risk Score", "needs_review": "Review?"})
display_df["Review?"] = display_df["Review?"].map({1: "⚠️ Yes", 0: "✅ No"})

st.subheader("Simulated Transactions")
st.dataframe(display_df, use_container_width=True, hide_index=True)

# =========================================================
# Step 1: Human Review
# =========================================================
st.subheader("Step 1: Review the uncertain cases")
st.markdown("Below are the cases where the model is **not confident**. For these cases, you have access to an extra investigation note.")

review_cases = df[df["needs_review"] == 1]
human_decisions = {}

if len(review_cases) == 0:
    st.info("The model is highly confident! No cases need review under current thresholds.")
else:
    cols = st.columns(2)
    for idx, (_, row) in enumerate(review_cases.iterrows()):
        with cols[idx % 2].expander(f"Review Case {int(row['case_id'])}"):
            st.write(f"**Amount:** ${row['amount']:,.2f} | **Risk Score:** {row['model_score']}")
            st.info(row['hidden_note'])
            decision = st.radio(f"Action for {int(row['case_id'])}:", ["Allow", "Block"], key=f"r_{row['case_id']}", horizontal=True)
            human_decisions[int(row['case_id'])] = 1 if decision == "Block" else 0

# =========================================================
# Step 2: Comparison
# =========================================================
st.divider()
st.subheader("Step 2: Compare System Performance")

# Apply decisions
df["hitl_pred"] = [human_decisions.get(int(r.case_id), 1 if r.model_score >= high_thr else 0) for _, r in df.iterrows()]

m_metrics = evaluate(df["model_only_pred"], df["true_label"])
h_metrics = evaluate(df["hitl_pred"], df["true_label"])

m_cost = expected_cost(df["model_only_pred"], df["true_label"], fn_cost=fn_cost, fp_cost=fp_cost)
h_cost = expected_cost(df["hitl_pred"], df["true_label"], df["needs_review"], fn_cost=fn_cost, fp_cost=fp_cost, review_cost=review_cost)

col1, col2 = st.columns(2)
with col1:
    st.metric("Model-Only Cost", f"${m_cost}")
    st.write(f"Accuracy: {m_metrics['Accuracy']}")
    st.write(f"Errors: {m_metrics['False Positives']} FP / {m_metrics['False Negatives']} FN")

with col2:
    st.metric("HITL System Cost", f"${h_cost}", delta=f"{m_cost - h_cost} Savings", delta_color="normal")
    st.write(f"Accuracy: {h_metrics['Accuracy']}")
    st.write(f"Errors: {h_metrics['False Positives']} FP / {h_metrics['False Negatives']} FN")

# =========================================================
# Step 3: Truth Reveal
# =========================================================
st.subheader("Step 3: Reveal Outcomes")
if st.checkbox("Show the actual ground truth for all cases"):
    reveal_df = df[["case_id", "model_score", "model_only_pred", "hitl_pred", "true_label"]].copy()
    reveal_df = reveal_df.rename(columns={"true_label": "Actual Fraud?"})
    st.dataframe(reveal_df, use_container_width=True, hide_index=True)

# =========================================================
# Final Takeaway
# =========================================================
st.divider()
st.subheader("Main Lesson")
st.markdown("""
A model can look strong on average and still be risky if it decides **every case by itself**. 

Effective human-AI systems focus on **routing the right decisions to the right decision maker**. In this simulation:
- The **Model** handles the high-volume, obvious cases.
- The **Human** handles the low-volume, high-ambiguity cases where external context (notes) changes the outcome.
""")