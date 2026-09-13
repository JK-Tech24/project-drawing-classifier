from tkinter import Tk, filedialog

import os
import re
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process

from project_drawing_classifier.tools.drawing_pdf_reader import (
    DrawingPDFReaderTool,
)


def main():
    # -------------------------------------------------
    # LOAD GEMINI API KEY
    # -------------------------------------------------

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    ENV_FILE = PROJECT_ROOT / ".env"
    load_dotenv(dotenv_path=ENV_FILE)

    if not os.getenv("GEMINI_API_KEY"):
        print("\nERROR: GEMINI_API_KEY was not found in .env")
        raise SystemExit(1)

    print("\nGemini API key loaded successfully.")


    # -------------------------------------------------
    # PDF FILE
    # -------------------------------------------------

    root = Tk()
    root.withdraw()

    PDF_PATH = filedialog.askopenfilename(
        title="Choose Drawing PDF",
        filetypes=[("PDF Files", "*.pdf")]
    )

    root.destroy()

    if not PDF_PATH:
        print("No PDF selected.")
        raise SystemExit


    # -------------------------------------------------
    # STEP 1 - READ PDF
    # -------------------------------------------------

    reader = DrawingPDFReaderTool()

    pdf_result = reader.run(
        file_path=PDF_PATH
    )

    print("\n================ PDF READER ================\n")
    print(pdf_result)

    if not pdf_result.startswith("PDF_READ_SUCCESS"):
        print("\nSTOPPED: PDF could not be read.")
        raise SystemExit(1)


    # -------------------------------------------------
    # STEP 2 - EXTRACT DRAWING INFORMATION
    # -------------------------------------------------

    extractor = Agent(
        role="Drawing Information Extractor",
        goal=(
            "Extract accurate structured information from construction "
            "and engineering drawing text."
        ),
        backstory=(
            "You specialize in architectural, structural, civil and MEP "
            "drawings. You only report information actually present in "
            "the drawing."
        ),
        llm="gemini/gemini-3.6-flash",
        verbose=True,
    )


    extract_task = Task(
        description=f"""
    Below is text extracted directly from an actual PDF drawing.

    Use ONLY this text.

    Do not infer information from the filename.
    Do not invent missing information.

    PDF CONTENT:

    {pdf_result}

    Extract:

    - Drawing Title
    - Drawing Number
    - Project Name
    - Sheet Number
    - Revision
    - Key Notes
    - Symbols / Labels
    - Discipline Indicators
    - Other Relevant Metadata

    Return exactly:

    Drawing Title:
    Drawing Number:
    Project Name:
    Sheet Number:
    Revision:
    Key Notes:
    Symbols / Labels:
    Discipline Indicators:
    Other Relevant Metadata:
    """,
        expected_output=(
            "Structured drawing information extracted from the PDF."
        ),
        agent=extractor,
    )


    # -------------------------------------------------
    # STEP 3 - CLASSIFY DRAWING
    # -------------------------------------------------

    classifier = Agent(
        role="Drawing Classification Specialist",
        goal=(
            "Classify construction drawings accurately into the correct discipline."
        ),
        backstory=(
            "You classify architectural, structural, MEP and civil drawings "
            "using drawing titles, drawing numbers, notes, labels and technical content."
        ),
        llm="gemini/gemini-3.6-flash",
        verbose=True,
    )


    classify_task = Task(
        description="""
    Using the drawing information produced by the previous task,
    classify the drawing into exactly ONE category:

    Architectural
    Structural
    MEP
    Civil
    Other

    Consider:

    - Drawing title
    - Drawing number / prefix
    - Title block
    - Technical terminology
    - Notes
    - Labels
    - Discipline-specific content

    Do not classify using only one isolated phrase.

    If conflicting information exists, weigh the strongest evidence,
    especially the explicit drawing title and drawing number.

    Confidence:

    95-100 = Explicit discipline with multiple strong indicators
    85-94  = Strong consistent evidence
    80-84  = Clear classification with minor ambiguity
    60-79  = Moderate evidence requiring human review
    Below 60 = Weak or conflicting evidence

    Return exactly:

    Drawing Title: <title>
    Drawing Number: <number>
    Discipline: <discipline>
    Confidence: <number>%
    Reason: <short evidence-based reason>
    """,
        expected_output=(
            "Drawing title, drawing number, discipline, confidence and reason."
        ),
        agent=classifier,
        context=[extract_task],
    )


    # -------------------------------------------------
    # STEP 4 - RUN CREWAI
    # -------------------------------------------------

    crew = Crew(
        agents=[
            extractor,
            classifier,
        ],
        tasks=[
            extract_task,
            classify_task,
        ],
        process=Process.sequential,
        verbose=True,
    )


    result = crew.kickoff()

    classification_result = str(result)


    # -------------------------------------------------
    # STEP 5 - READ CLASSIFICATION RESULT
    # -------------------------------------------------

    def get_field(text, field_name):
        pattern = rf"{re.escape(field_name)}:\s*(.+)"
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            return match.group(1).strip()

        return "Not Found"


    drawing_title = get_field(
        classification_result,
        "Drawing Title"
    )

    drawing_number = get_field(
        classification_result,
        "Drawing Number"
    )

    discipline = get_field(
        classification_result,
        "Discipline"
    )

    confidence_text = get_field(
        classification_result,
        "Confidence"
    )

    reason = get_field(
        classification_result,
        "Reason"
    )


    # -------------------------------------------------
    # STEP 6 - CONVERT CONFIDENCE TO NUMBER
    # -------------------------------------------------

    confidence_match = re.search(
        r"(\d+)",
        confidence_text
    )

    if confidence_match:
        confidence = int(
            confidence_match.group(1)
        )
    else:
        confidence = 0


    # -------------------------------------------------
    # STEP 7 - CONFIDENCE ROUTER
    # -------------------------------------------------

    if confidence >= 80:

        routing_decision = (
            f"Auto-route to {discipline} workflow"
        )

        human_review_required = "No"
        reviewer_name = "N/A"
        final_discipline = discipline
        final_action = "Automatically Routed"

    else:

        routing_decision = "Send for Human Review"

        human_review_required = "Yes"

        print(
            "\n================ HUMAN REVIEW REQUIRED ================\n"
        )

        print(
            f"AI Classification: {discipline}"
        )

        print(
            f"Confidence: {confidence}%"
        )

        print(
            f"Reason: {reason}"
        )

        reviewer_name = input(
            "\nReviewer Name: "
        ).strip()

        final_discipline = input(
            "Enter Final Discipline "
            "(Architectural / Structural / MEP / Civil / Other): "
        ).strip()

        if not final_discipline:
            final_discipline = discipline

        final_action = "Human Reviewed and Routed"


    # -------------------------------------------------
    # STEP 8 - CREATE AUDIT RECORD
    # -------------------------------------------------

    date_time = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


    audit_record = f"""
    ================ FINAL AUDIT RECORD ================

    Drawing Title: {drawing_title}
    Drawing Number: {drawing_number}
    Discipline: {discipline}
    Confidence: {confidence}%
    Classification Reason: {reason}
    Routing Decision: {routing_decision}
    Human Review Required: {human_review_required}
    Reviewer Name: {reviewer_name}
    Final Discipline: {final_discipline}
    Date/Time: {date_time}
    Final Action: {final_action}

    ====================================================
    """


    # -------------------------------------------------
    # STEP 9 - DISPLAY FINAL RESULT
    # -------------------------------------------------

    print(
        "\n================ CLASSIFICATION RESULT ================\n"
    )

    print(
        classification_result
    )

    print(
        audit_record
    )


    # -------------------------------------------------
    # STEP 10 - SAVE AUDIT RECORD
    # -------------------------------------------------

    audit_file = Path(__file__).with_name(
        "drawing_classification_audit.txt"
    )

    with open(
        audit_file,
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            audit_record
        )

        file.write(
            "\n"
        )


    print(
        f"\nAudit record saved to:\n{audit_file}"
    )

if __name__ == "__main__":
    main()
