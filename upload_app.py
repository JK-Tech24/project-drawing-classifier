import base64
import os
import time

import requests
import streamlit as st
from dotenv import load_dotenv


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

load_dotenv(override=True)

CREWAI_URL = os.getenv(
    "CREWAI_URL",
    "https://project-drawing-classifier-7b38518f-ab2d-42-33e4f1d2.crewai.com",
).rstrip("/")

CREWAI_API_TOKEN = os.getenv("CREWAI_API_TOKEN", "")


# ---------------------------------------------------------
# Page
# ---------------------------------------------------------

st.set_page_config(
    page_title="Project Drawing Classifier",
    page_icon="📐",
    layout="centered",
)

st.title("Project Drawing Classifier")

st.write(
    "Upload a construction or engineering drawing PDF. "
    "CrewAI will read the drawing, classify its discipline, "
    "apply the confidence rule, and return the routing decision."
)


# ---------------------------------------------------------
# Check configuration
# ---------------------------------------------------------

if not CREWAI_API_TOKEN:
    st.warning(
        "CrewAI API token is not configured. "
        "Add CREWAI_API_TOKEN to the .env file."
    )


# ---------------------------------------------------------
# Upload PDF
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "Choose Drawing PDF",
    type=["pdf"],
    accept_multiple_files=False,
)


if uploaded_file is not None:

    st.success(f"Selected: {uploaded_file.name}")

    pdf_bytes = uploaded_file.getvalue()

    st.caption(
        f"File size: {len(pdf_bytes) / 1024:.1f} KB"
    )

    if st.button(
        "Classify Drawing",
        type="primary",
        use_container_width=True,
    ):

        if not CREWAI_API_TOKEN:
            st.error(
                "CREWAI_API_TOKEN is missing from the .env file."
            )
            st.stop()

        # Convert PDF to Base64 internally
        pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")

        payload = {
            "inputs": {
                "drawing_filename": uploaded_file.name,
                "drawing_pdf_b64": pdf_base64,
            }
        }

        headers = {
            "Authorization": f"Bearer {CREWAI_API_TOKEN}",
            "Content-Type": "application/json",
        }

        try:

            # -------------------------------------------------
            # Start CrewAI Flow
            # -------------------------------------------------

            with st.spinner("Sending drawing to CrewAI..."):

                kickoff_response = requests.post(
                    f"{CREWAI_URL}/kickoff",
                    headers=headers,
                    json=payload,
                    timeout=120,
                )

            if not kickoff_response.ok:

                st.error(
                    f"CrewAI kickoff failed "
                    f"({kickoff_response.status_code})."
                )

                st.code(kickoff_response.text)

                st.stop()

            kickoff_data = kickoff_response.json()

            kickoff_id = (
                kickoff_data.get("kickoff_id")
                or kickoff_data.get("id")
                or kickoff_data.get("execution_id")
            )

            if not kickoff_id:

                st.error(
                    "CrewAI started the request but no kickoff ID "
                    "was found in the response."
                )

                st.json(kickoff_data)
                st.stop()

            st.info("CrewAI is processing the drawing...")

            # -------------------------------------------------
            # Poll CrewAI status
            # -------------------------------------------------

            status_placeholder = st.empty()

            final_data = None

            for attempt in range(120):

                status_response = requests.get(
                    f"{CREWAI_URL}/status/{kickoff_id}",
                    headers=headers,
                    timeout=60,
                )

                if not status_response.ok:

                    st.error(
                        f"Could not retrieve CrewAI status "
                        f"({status_response.status_code})."
                    )

                    st.code(status_response.text)
                    st.stop()

                status_data = status_response.json()

                # CrewAI uses "state"
                state_value = (
                    status_data.get("state")
                    or status_data.get("status")
                    or "Processing"
                )

                state = str(state_value).lower()

                status_placeholder.write(
                    f"Status: {state_value}"
                )

                if state in {
                    "success",
                    "successful",
                    "completed",
                    "complete",
                    "finished",
                }:
                    final_data = status_data
                    break

                if state in {
                    "failure",
                    "failed",
                    "error",
                    "cancelled",
                    "canceled",
                }:
                    st.error("CrewAI processing failed.")
                    st.json(status_data)
                    st.stop()

                time.sleep(2)

            if final_data is None:
                st.error(
                    "The request is taking longer than expected."
                )
                st.stop()

            status_placeholder.empty()

            # -------------------------------------------------
            # Show result
            # -------------------------------------------------

            st.success("Drawing classification completed.")

            result = (
                final_data.get("result")
                or final_data.get("output")
                or final_data.get("response")
                or final_data.get("data")
                or final_data
            )

            st.subheader("Classification Result")

            if isinstance(result, str):

                st.text(result)

            elif isinstance(result, dict):

                fields = [
                    ("Drawing Title", "drawing_title"),
                    ("Drawing Number", "drawing_number"),
                    ("Discipline", "discipline"),
                    ("Confidence", "confidence"),
                    ("Classification Reason", "reason"),
                    ("Routing Decision", "routing_decision"),
                    (
                        "Human Review Required",
                        "human_review_required",
                    ),
                    ("Reviewer Name", "reviewer_name"),
                    ("Final Discipline", "final_discipline"),
                    ("Final Action", "final_action"),
                ]

                found_fields = False

                for label, key in fields:

                    if key in result:

                        found_fields = True

                        value = result[key]

                        if key == "confidence":
                            try:
                                value = f"{float(value):.0f}%"
                            except Exception:
                                pass

                        st.write(
                            f"**{label}:** {value}"
                        )

                if not found_fields:
                    st.json(result)

            else:
                st.write(result)

            # -------------------------------------------------
            # Technical response
            # -------------------------------------------------

            with st.expander("Technical Response"):
                st.json(final_data)

        except requests.exceptions.RequestException as exc:

            st.error(
                "Could not connect to the CrewAI deployment."
            )

            st.code(str(exc))

        except Exception as exc:

            st.error("Unexpected error.")

            st.code(str(exc))