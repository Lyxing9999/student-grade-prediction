"""Streamlit interface for student grade prediction."""

import pandas as pd
import streamlit as st

from src.prediction_service import (
    GradePredictionService,
)


st.set_page_config(
    page_title="Student Grade Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        .block-container {
            max-width: 1100px;
            padding-top: 2.5rem;
            padding-bottom: 3rem;
        }

        .hero {
            padding: 1.8rem 2rem;
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 18px;
            margin-bottom: 1.5rem;
        }

        .hero h1 {
            margin: 0;
            font-size: 2.25rem;
        }

        .hero p {
            margin: 0.6rem 0 0;
            opacity: 0.72;
            font-size: 1.05rem;
        }

        .result-card {
            padding: 1.5rem;
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 16px;
            margin-top: 1rem;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.18);
            padding: 1rem;
            border-radius: 14px;
        }

        .small-note {
            opacity: 0.68;
            font-size: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_prediction_service() -> GradePredictionService:
    """Load the model once for the Streamlit session."""

    return GradePredictionService()


st.markdown(
    """
    <div class="hero">
        <h1>🎓 Student Grade Prediction</h1>
        <p>
            Predict grade A, B, C, D, or F using student
            study behavior and classroom information.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


try:
    service = load_prediction_service()
except FileNotFoundError as error:
    st.error("The trained model is not available.")

    st.code(
        "python -m scripts.train_final_model",
        language="bash",
    )

    st.caption(str(error))
    st.stop()
except Exception as error:
    st.error("The saved model could not be loaded.")
    st.exception(error)
    st.stop()


with st.sidebar:
    st.header("Model Information")

    st.write(f"**{service.model_name}**")

    metrics = service.metrics

    if metrics:
        st.metric(
            "Macro F1",
            f"{metrics.get('macro_f1', 0):.4f}",
        )

        st.metric(
            "Accuracy",
            f"{metrics.get('accuracy', 0):.4f}",
        )

    st.divider()

    st.write("**Why this model?**")

    st.caption(
        "Random Forest achieved the highest tuned "
        "Macro F1-score among the four compared models."
    )

    st.divider()

    st.caption(
        "Educational demonstration using a synthetic "
        "student-performance dataset."
    )


st.subheader("Enter student information")

with st.form("student_prediction_form"):
    first_column, second_column, third_column = (
        st.columns(3)
    )

    with first_column:
        weekly_self_study_hours = st.number_input(
            "Weekly self-study hours",
            min_value=0.0,
            max_value=80.0,
            value=15.0,
            step=0.5,
            help=(
                "Independent study hours completed "
                "during one week."
            ),
        )

    with second_column:
        attendance_percentage = st.number_input(
            "Attendance percentage",
            min_value=0.0,
            max_value=100.0,
            value=85.0,
            step=1.0,
            help="Percentage of classes attended.",
        )

    with third_column:
        class_participation = st.number_input(
            "Class participation",
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.5,
            help=(
                "Participation score based on classroom "
                "activities and discussions."
            ),
        )

    submitted = st.form_submit_button(
        "Predict student grade",
        type="primary",
        use_container_width=True,
    )


if submitted:
    try:
        prediction = service.predict(
            weekly_self_study_hours=(
                weekly_self_study_hours
            ),
            attendance_percentage=(
                attendance_percentage
            ),
            class_participation=(
                class_participation
            ),
        )

        highest_probability = max(
            prediction.probabilities.values()
        )

        st.markdown(
            '<div class="result-card">',
            unsafe_allow_html=True,
        )

        st.subheader("Prediction Result")

        result_column, probability_column = (
            st.columns(2)
        )

        with result_column:
            st.metric(
                "Predicted Grade",
                prediction.predicted_grade,
            )

        with probability_column:
            st.metric(
                "Highest model probability",
                f"{highest_probability:.1%}",
            )

        st.markdown("</div>", unsafe_allow_html=True)

        probability_table = pd.DataFrame(
            {
                "Grade": list(
                    prediction.probabilities.keys()
                ),
                "Probability": list(
                    prediction.probabilities.values()
                ),
            }
        )

        st.subheader("Probability by Grade")

        chart_data = probability_table.set_index(
            "Grade"
        )

        st.bar_chart(
            chart_data,
            y="Probability",
        )

        display_table = probability_table.copy()

        display_table["Probability"] = (
            display_table["Probability"]
            .map(lambda value: f"{value:.2%}")
        )

        st.dataframe(
            display_table,
            hide_index=True,
            use_container_width=True,
        )

        st.info(
            "The probability values are model estimates. "
            "They should not be treated as an official "
            "academic assessment."
        )

    except Exception as error:
        st.error(
            "The prediction could not be completed."
        )
        st.exception(error)


with st.expander("How the prediction works"):
    st.markdown(
        """
        1. Enter the student's three input features.
        2. The interface sends the values to the saved model.
        3. Random Forest combines predictions from 150 trees.
        4. The grade with the strongest combined result is shown.
        """
    )


with st.expander("Project limitations"):
    st.markdown(
        """
        - The dataset is synthetic.
        - The grade classes are highly imbalanced.
        - Grade F is difficult to predict precisely.
        - Weekly study hours provides most of the useful signal.
        - The result is for educational demonstration only.
        """
    )
