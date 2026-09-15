"""
BreastPredict AI -- AI-Assisted Breast Cytology Decision Support System
Department of Pathology, Atal Bihari Vajpayee Government Medical College, Vidisha

Phase 1.5: adds outcome recording so real Robinson-params -> confirmed-
histopathology training data accumulates over time (see "Record Outcome"
tab). The Predict tab's risk tier remains an explicitly-labeled
placeholder lookup, not a trained model -- see comment above
placeholder_risk_tier() below.
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


INTAKE_LOG_PATH = os.path.join(os.path.dirname(__file__), "case_log.csv")
OUTCOME_LOG_PATH = os.path.join(os.path.dirname(__file__), "outcome_log.csv")

INTAKE_FIELDNAMES = [
    "timestamp_utc", "case_label", "age", "side", "axillary_lymphadenopathy",
    "birads_category", "cell_dissociation", "cell_size", "cell_uniformity",
    "nucleoli", "nuclear_margin", "chromatin", "robinson_score", "robinson_grade",
    "placeholder_risk_tier",
]

OUTCOME_FIELDNAMES = [
    "timestamp_utc", "case_label", "confirmed_nottingham_grade",
    "confirmed_diagnosis", "confirmed_malignancy",
]


def append_row(path, fieldnames, record: dict) -> None:
    file_exists = os.path.exists(path)
    record = {**record, "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: record.get(k, "") for k in fieldnames})


def read_rows(path, fieldnames) -> list:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def case_label_exists_in_intake(case_label: str) -> bool:
    rows = read_rows(INTAKE_LOG_PATH, INTAKE_FIELDNAMES)
    return any(r.get("case_label", "").strip() == case_label.strip() for r in rows)


st.set_page_config(page_title="BreastPredict AI", page_icon="🩺", layout="centered")

st.title("BreastPredict AI")
st.caption("AI-Assisted Breast Cytology Decision Support System")
st.markdown("**Department of Pathology** · Atal Bihari Vajpayee Government Medical College, Vidisha")
st.divider()

tab_predict, tab_outcome = st.tabs(["🆕 New Case", "📋 Record Outcome"])

with tab_predict:
    st.header("1. Patient Information")
    col1, col2 = st.columns(2)
    with col1:
        case_label = st.text_input(
            "Case label (not a patient name)",
            help="Use a study ID / case number -- this app does not store patient-identifying "
                 "information. Remember this label -- you'll need it later to record the outcome.",
            key="intake_case_label",
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
        if not case_label.strip():
            st.error("Please enter a case label before predicting -- you'll need it to record the outcome later.")
        else:
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
            st.info("Not available -- requires a trained model on confirmed histopathology outcomes. "
                    "Use the 'Record Outcome' tab once this case's histopathology is back, to help build that dataset.")

            st.subheader("Likely Diagnosis")
            st.info("Not available -- diagnostic terminology must come from the trained model's own dataset (Phase 2).")

            try:
                append_row(INTAKE_LOG_PATH, INTAKE_FIELDNAMES, {
                    "case_label": case_label, "age": age, "side": side,
                    "axillary_lymphadenopathy": axillary_lymphadenopathy,
                    "birads_category": birads_category, "cell_dissociation": cell_dissociation,
                    "cell_size": cell_size, "cell_uniformity": cell_uniformity, "nucleoli": nucleoli,
                    "nuclear_margin": nuclear_margin, "chromatin": chromatin,
                    "robinson_score": robinson_result.score, "robinson_grade": robinson_result.grade,
                    "placeholder_risk_tier": risk_result.tier,
                })
                st.caption(f"✅ Case '{case_label}' logged. Remember this label for the outcome step later.")
            except Exception as e:
                st.caption(f"⚠️ Could not log case: {e}")

with tab_outcome:
    st.header("Record Confirmed Histopathology Outcome")
    st.caption(
        "Once a case's real histopathology result is back, record it here using the same "
        "case label you used at intake. This links the intake inputs (BI-RADS + Robinson "
        "parameters) to the real outcome -- the dataset a future model will actually be "
        "trained on."
    )

    outcome_case_label = st.text_input("Case label (must match the one used at intake)", key="outcome_case_label")

    if outcome_case_label.strip():
        if case_label_exists_in_intake(outcome_case_label):
            st.success(f"Found a matching intake record for '{outcome_case_label}'.")
        else:
            st.warning(
                f"No intake record found for '{outcome_case_label}' in this app session's log yet. "
                "You can still save the outcome -- it'll link up once the matching intake case exists in the log."
            )

    confirmed_malignancy = st.selectbox("Confirmed outcome", ["Benign", "Malignant"])
    confirmed_nottingham_grade = st.selectbox(
        "Confirmed Nottingham Grade (leave as N/A if not malignant)",
        ["N/A", "Grade I", "Grade II", "Grade III"],
    )
    confirmed_diagnosis = st.text_input("Confirmed histopathology diagnosis (free text)")

    if st.button("💾 Save Outcome", type="primary", use_container_width=True):
        if not outcome_case_label.strip():
            st.error("Please enter the case label.")
        elif not confirmed_diagnosis.strip():
            st.error("Please enter the confirmed diagnosis text.")
        else:
            try:
                append_row(OUTCOME_LOG_PATH, OUTCOME_FIELDNAMES, {
                    "case_label": outcome_case_label,
                    "confirmed_nottingham_grade": confirmed_nottingham_grade,
                    "confirmed_diagnosis": confirmed_diagnosis,
                    "confirmed_malignancy": confirmed_malignancy,
                })
                st.success(f"✅ Outcome for '{outcome_case_label}' saved.")
            except Exception as e:
                st.error(f"Could not save outcome: {e}")

    st.divider()
    outcomes_so_far = read_rows(OUTCOME_LOG_PATH, OUTCOME_FIELDNAMES)
    st.caption(f"Outcomes recorded so far (this session): {len(outcomes_so_far)}")

st.divider()
st.caption(
    "🎓 Research/Academic Prototype. This tool supports — and does not replace — "
    "professional pathological and clinical judgment. Any AI-generated prediction must be "
    "appropriately trained, internally validated and, where applicable, externally validated "
    "before consideration for clinical use."
            )
