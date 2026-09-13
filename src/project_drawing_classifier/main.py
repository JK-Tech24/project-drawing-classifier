from tkinter import Tk, filedialog

import os
import re
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from pydantic import BaseModel

from crewai import Agent, Task, Crew, Process
from crewai.flow import Flow, start, listen

from project_drawing_classifier.tools.drawing_pdf_reader import (
    DrawingPDFReaderTool,
)


# -------------------------------------------------
# FLOW STATE
# -------------------------------------------------

class DrawingClassifierState(BaseModel):
    pdf_path: str = ""
    pdf_text: str = ""

    drawing_title: str = ""
    drawing_number: str = ""
    discipline: str = ""
    confidence: int = 0
    reason: str = ""

    routing_decision: str = ""
    human_review_required: str = ""
    reviewer_name: str = ""
    final_discipline: str = ""
    final_action: str = ""

    audit_record: str = ""


# -------------------------------------------------
# CREWAI FLOW
# -------------------------------------------------

class ProjectDrawingClassifierFlow(Flow[DrawingClassifierState]):

    @start()
    def choose_and_read_pdf(self):

        # -----------------------------------------
        # LOAD ENVIRONMENT
        # -----------------------------------------

        project_root = Path(__file__).resolve().parents[2]
        env_file = project_root / ".env"

        load_dotenv(
            dotenv_path=env_file,
            override=True,
        )

        if not os.getenv("GEMINI_API_KEY"):
            raise RuntimeError(
                "GEMINI_API_KEY was not found in .env"
            )

        print("\nGemini API key loaded successfully.")

        # -----------------------------------------
        # CHOOSE PDF
        # -----------------------------------------

        root = Tk()
        root.withdraw()

        pdf_path = filedialog.askopenfilename(
            title="Choose Drawing PDF",
            filetypes=[
                ("PDF Files", "*.pdf")
            ],
        )

        root.destroy()

        if not pdf_path:
            raise RuntimeError(
                "No PDF selected."
            )

        self.state.pdf_path = pdf_path

        # -----------------------------------------
        # READ PDF USING CUSTOM CREWAI TOOL
        # -----------------------------------------

        reader = DrawingPDFReaderTool()

        pdf_result = reader.run(
            file_path=pdf_path
        )

        print(
            "\n================ PDF READER ================\n"
        )

        print(pdf_result)

        if not pdf_result.startswith(
            "PDF_READ_SUCCESS"
        ):
            raise RuntimeError(
                "PDF could not be read."
            )

        self.state.pdf_text = pdf_result

        return pdf_result


    @listen(choose_and_read_pdf)
    def classify_drawing(self, pdf_result):

        # -----------------------------------------
        # INFORMATION EXTRACTOR AGENT
        # -----------------------------------------

        extractor = Agent(

            role="Drawing Information Extractor",

            goal=(
                "Extract accurate drawing information "
                "from the text read directly from the PDF drawing."
            ),

            backstory=(
                "You are experienced in reading construction "
                "and engineering drawings, title blocks, "
                "drawing numbers, revisions, notes, symbols, "
                "labels and discipline indicators."
            ),

            llm="gemini/gemini-3.6-flash",

            verbose=True,
        )

        # -----------------------------------------
        # EXTRACTION TASK
        # -----------------------------------------

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
                "Structured drawing information "
                "extracted only from the supplied PDF text."
            ),

            agent=extractor,
        )

        # -----------------------------------------
        # CLASSIFICATION AGENT
        # -----------------------------------------

        classifier = Agent(

            role="Drawing Classification Specialist",

            goal=(
                "Classify construction and engineering drawings "
                "accurately into Architectural, Structural, "
                "MEP, Civil, or Other."
            ),

            backstory=(
                "You are a senior engineering drawing reviewer "
                "familiar with architectural, structural, civil "
                "and MEP drawing conventions, drawing-number "
                "prefixes, title blocks and technical content."
            ),

            llm="gemini/gemini-3.6-flash",

            verbose=True,
        )

        # -----------------------------------------
        # CLASSIFICATION TASK
        # -----------------------------------------

        classification_task = Task(

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
                "Drawing Title, Drawing Number, Discipline, "
                "Confidence percentage and classification reason."
            ),

            agent=classifier,

            context=[
                extract_task
            ],
        )

        # -----------------------------------------
        # RUN CREW
        # -----------------------------------------

        crew = Crew(

            agents=[
                extractor,
                classifier,
            ],

            tasks=[
                extract_task,
                classification_task,
            ],

            process=Process.sequential,

            verbose=True,
        )

        result = crew.kickoff()

        classification_result = str(result)

        print(
            "\n================ CLASSIFICATION RESULT ================\n"
        )

        print(classification_result)

        # -----------------------------------------
        # PARSE RESULT
        # -----------------------------------------

        def get_field(text, field_name):

            pattern = (
                rf"{re.escape(field_name)}:\s*(.+)"
            )

            match = re.search(
                pattern,
                text,
                re.IGNORECASE,
            )

            if match:
                return match.group(1).strip()

            return "Not Found"


        self.state.drawing_title = get_field(
            classification_result,
            "Drawing Title",
        )

        self.state.drawing_number = get_field(
            classification_result,
            "Drawing Number",
        )

        self.state.discipline = get_field(
            classification_result,
            "Discipline",
        )

        confidence_text = get_field(
            classification_result,
            "Confidence",
        )

        self.state.reason = get_field(
            classification_result,
            "Reason",
        )

        confidence_match = re.search(
            r"(\d+)",
            confidence_text,
        )

        if confidence_match:
            self.state.confidence = int(
                confidence_match.group(1)
            )
        else:
            self.state.confidence = 0

        return classification_result


    @listen(classify_drawing)
    def route_by_confidence(self):

        # -----------------------------------------
        # ROUTING RULE
        # -----------------------------------------

        if self.state.confidence >= 80:

            self.state.routing_decision = (
                f"Auto-route to "
                f"{self.state.discipline} workflow"
            )

            self.state.human_review_required = "No"

            self.state.reviewer_name = "N/A"

            self.state.final_discipline = (
                self.state.discipline
            )

            self.state.final_action = (
                "Automatically Routed"
            )

        else:

            self.state.routing_decision = (
                "Send for Human Review"
            )

            self.state.human_review_required = "Yes"

            print(
                "\n================ "
                "HUMAN REVIEW REQUIRED "
                "================\n"
            )

            print(
                f"AI Classification: "
                f"{self.state.discipline}"
            )

            print(
                f"Confidence: "
                f"{self.state.confidence}%"
            )

            print(
                f"Reason: "
                f"{self.state.reason}"
            )

            reviewer_name = input(
                "\nReviewer Name: "
            ).strip()

            final_discipline = input(
                "\nEnter Final Discipline "
                "(Architectural / Structural / "
                "MEP / Civil / Other): "
            ).strip()

            if not reviewer_name:
                reviewer_name = "Not Provided"

            if not final_discipline:
                final_discipline = (
                    self.state.discipline
                )

            self.state.reviewer_name = (
                reviewer_name
            )

            self.state.final_discipline = (
                final_discipline
            )

            self.state.final_action = (
                "Human Reviewed and Routed"
            )


    @listen(route_by_confidence)
    def create_audit_record(self):

        # -----------------------------------------
        # CREATE FINAL AUDIT RECORD
        # -----------------------------------------

        date_time = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        audit_record = f"""
================ FINAL AUDIT RECORD ================

Drawing Title: {self.state.drawing_title}
Drawing Number: {self.state.drawing_number}
Discipline: {self.state.discipline}
Confidence: {self.state.confidence}%
Classification Reason: {self.state.reason}
Routing Decision: {self.state.routing_decision}
Human Review Required: {self.state.human_review_required}
Reviewer Name: {self.state.reviewer_name}
Final Discipline: {self.state.final_discipline}
Date/Time: {date_time}
Final Action: {self.state.final_action}

====================================================
"""

        self.state.audit_record = audit_record

        print(audit_record)

        # -----------------------------------------
        # SAVE AUDIT RECORD
        # -----------------------------------------

        audit_file = Path(__file__).with_name(
            "drawing_classification_audit.txt"
        )

        with open(
            audit_file,
            "a",
            encoding="utf-8",
        ) as file:

            file.write(audit_record)
            file.write("\n")

        print(
            f"\nAudit record saved to:\n"
            f"{audit_file}"
        )

        return audit_record


# -------------------------------------------------
# CREWAI CLI ENTRY POINT
# -------------------------------------------------

def kickoff():

    flow = ProjectDrawingClassifierFlow()

    return flow.kickoff()


def plot():

    flow = ProjectDrawingClassifierFlow()

    return flow.plot(
        "ProjectDrawingClassifierFlow"
    )


# -------------------------------------------------
# CREWAI CLI ENTRY POINT
# -------------------------------------------------

def kickoff():
    flow = ProjectDrawingClassifierFlow()
    return flow.kickoff()


def plot():
    flow = ProjectDrawingClassifierFlow()
    return flow.plot(
        "ProjectDrawingClassifierFlow"
    )


if __name__ == "__main__":
    kickoff()