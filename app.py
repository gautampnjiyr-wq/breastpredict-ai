"""
BreastPredict AI -- AI-Assisted Breast Cytology Decision Support System
Department of Pathology, Atal Bihari Vajpayee Government Medical College, Vidisha
"""

import csv
import os
from dataclasses import dataclass
from datetime import datetime, timezone

import streamlit as st

CANONICAL_CATEGORIES = ["2", "3", "4A", "4B", "4C", "5"]

CATEGORY_DESCRIPTIONS = {
    "2": "Benign",
    "3": "Probably benign",
    "4A": "Low suspicion for malignancy",
    "4B": "Moderate suspicion for malignancy",
    "4C": "High suspicion for malignancy",
    "5": "Highly suggestive of malignancy",
}

CELL_DISSOCIATION_OPTIONS = {
    "Mostly Clusters": 1,
    "Mixed Clusters and Single Cells": 2,
    "Mostly Single Cells": 3,
}


@dataclass
class RobinsonInput:
    cell_dissociation: str
    cell_size: int
    cell_uniformity: int
    nucleoli: int
    nuclear_margin: int
    chromatin: int


@dataclass
class RobinsonResult:
    score: int
    grade: str


def score_robinson(data: RobinsonInput) -> RobinsonResult:
    dissociation_score = CELL_DISSOCIATION_OPTIONS[data.cell_dissociation]
    total = (
        dissociation_score
        + data.cell_size
        + data.cell_uniformity
        + data.nucleoli
        + data.nuclear_margin
        + data.chromatin
    )
    if total <= 11:
        grade = "Grade I"
    elif total <= 14:
        grade = "Grade II"
    else:
        grade = "Grade III"
    return RobinsonResult(score=total, grade=grade)


_BIRADS_TIER = {
    "2": "low", "3": "low",
    "4A": "intermediate", "4B": "intermediate",
    "4C": "high", "5": "high",
}

_COMBINATION_TABLE = {
    ("low", "Grade I"): "Low",
    ("low", "Grade II"): "Low-Intermediate",
    ("low", "Grade III"): "Intermediate",
    ("intermediate", "Grade I"): "Low-Intermediate",
    ("intermediate", "Grade II"): "Intermediate",
    ("intermediate", "Grade III"): "Intermediate-High",
    ("high", "Grade I"): "Intermediate",
    ("high", "Grade II"): "Intermediate-High",
    ("high", "Grade III"): "High",
}


@dataclass
class PlaceholderRiskResult:
    tier: str
    basis: str


def placeholder_risk_tier(birads_category: str, robinson_grade: str) -> PlaceholderRiskResult:
    birads_tier = _BIRADS_TIER[birads_category]
    tier = _COMBINATION_TABLE[(birads_tier, robinson_grade)]
    return PlaceholderRiskResult(
        tier=tier,
        basis=f"BI-RADS {birads_category} + Robinson {robinson_grade} looked up in a fixed table (not a trained model).",
    )


CASE_LOG_PATH = os.path.join(os.path.dirname(__file__), "case_log.csv")
FIELDNAMES = [
    "timestamp_utc", "case_label", "age", "side", "axillary_lymphadenopathy",
    "birads_category", "cell_dissociation", "cell_size", "cell_uniformity",
    "nucleoli", "nuclear_margin", "chromatin", "robinson_score", "robinson_grade",
    "placeholder_risk_tier",
]


def append_case(record: dict) -> None:
    file_exists = os.path.exists(CASE_LOG_PATH)
    record = {**record, "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    with open(CASE_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: record.get(k, "") for k in FIELDNAMES})


st.set_page_config(page_title="BreastPredict AI", page_icon="🩺", layout="centered")

st.title("BreastPredict AI")
st.caption("AI-Assisted Breast Cytology Decision Support System")
st.markdown("**Department of Pathology** · Atal Bihari Vajpayee Government Medical College, Vidisha")
st.divider()

st.header("1. Patient Information")
col1, col2 = st.columns(2)
with col1:
    case_label = st.text_input(
        "Case label (not a patient name)",
        help="Use a study ID / case number -- this app does not store patient-identifying information.",
    )
    age = st.number_input("Age", min_value=0, max_value=120, value=40, step=1)
with col2:
    side = st.selectbox("Site / Side of Breast", ["Right breast", "Left breast"])
    axillary_lymphadenopathy = st.selectbox("Axillary Lymphadenopathy", ["Absent", "Present"])

st.header("2. Radiology")
birads_category = st.selectbox(
    "BI-RADS Category", CANONICAL_CATEGORIES,
    format_func=lambda c: f"BI-RADS {c} — {CATEGORY_DESCRIPTIONS[c]}",
)

st.header("3. Robinson Cytological Grading")
cell_dissociation = st.selectbox("Cell Dissociation", list(CELL_DISSOCIATION_OPTIONS.keys()))
rcol1, rcol2, rcol3, rcol4, rcol5 = st.columns(5)
with rcol1:
    cell_size = st.selectbox("Cell Size", [1, 2, 3])
with rcol2:
    cell_uniformity = st.selectbox("Cell Uniformity", [1, 2, 3])
with rcol3:
    nucleoli = st.selectbox("Nucleoli", [1, 2, 3])
with rcol4:
    nuclear_margin = st.selectbox("Nuclear Margin", [1, 2, 3])
with rcol5:
    chromatin = st.selectbox("Chromatin", [1, 2, 3])

st.divider()

if st.button("🔍 Predict", type="primary", use_container_width=True):
    robinson_result = score_robinson(RobinsonInput(
        cell_dissociation, cell_size, cell_uniformity, nucleoli, nuclear_margin, chromatin
    ))
    risk_result = placeholder_risk_tier(birads_category, robinson_result.grade)

    st.header("4. Robinson Score and Grade")
    m1, m2 = st.columns(2)
    m1.metric("Robinson Score", f"{robinson_result.score} / 18")
    m2.metric("Robinson Grade", robinson_result.grade)

    st.header("5. AI Prediction")
    st.warning(
        "⚠️ No trained predictive model exists yet. The tier below is a fixed, documented "
        "lookup of BI-RADS category + Robinson grade -- **not** a probability, and not "
        "clinically validated."
    )
    st.metric("Preliminary Risk Tier (placeholder)", risk_result.tier)
    st.caption(risk_result.basis)

    st.subheader("Predicted Nottingham Grade")
    st.info("Not available -- requires a trained model on validated histopathology outcomes (Phase 2).")

    st.subheader("Likely Diagnosis")
    st.info("Not available -- diagnostic terminology must come from the trained model's own dataset (Phase 2).")

    try:
        append_case({
            "case_label": case_label, "age": age, "side": side,
            "axillary_lymphadenopathy": axillary_lymphadenopathy,
            "birads_category": birads_category, "cell_dissociation": cell_dissociation,
            "cell_size": cell_size, "cell_uniformity": cell_uniformity, "nucleoli": nucleoli,
            "nuclear_margin": nuclear_margin, "chromatin": chromatin,
            "robinson_score": robinson_result.score, "robinson_grade": robinson_result.grade,
            "placeholder_risk_tier": risk_result.tier,
        })
        st.caption("✅ Case logged (session-local; see note in code about durable storage).")
    except Exception as e:
        st.caption(f"⚠️ Could not log case: {e}")

st.divider()
st.caption(
    "🎓 Research/Academic Prototype. This tool supports — and does not replace — "
    "professional pathological and clinical judgment. Any AI-generated prediction must be "
    "appropriately trained, internally validated and, where applicable, externally validated "
    "before consideration for clinical use."
)
