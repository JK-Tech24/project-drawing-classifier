import base64
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, PrivateAttr

from crewai import Agent, Task, Crew, Process
from crewai.flow import Flow, start, listen

from project_drawing_classifier.tools.drawing_pdf_reader import (
    DrawingPDFReaderTool,
)


# -------------------------------------------------
# FLOW STATE
# -------------------------------------------------

class DrawingClassifierState(BaseModel):
    # Only these two fields are exposed by AMP as Flow inputs.
    drawing_filename: str = ""
    drawing_pdf_b64: str = ""

    # Everything else stays internal to the running Flow.
    _internal: dict = PrivateAttr(default_factory=dict)


# -------------------------------------------------
# CREWAI FLOW
# -------------------------------------------------

class ProjectDrawingClassifierFlow(Flow[DrawingClassifierState]):

    @start()
    def receive_and_read_pdf(self):

        # -----------------------------------------
        # LOAD ENVIRONMENT
        # -----------------------------------------

        project_root = Path(__file__).resolve().parents[2]
        env_file = project_root / ".env"

        load_dotenv(
            dotenv_path=env_file,
            override=False,
        )

        if not os.getenv("GEMINI_API_KEY"):
            raise RuntimeError(
                "GEMINI_API_KEY was not found in the environment."
            )

        print("\nGemini API key loaded successfully.")

        # -----------------------------------------
        # VALIDATE INPUTS
        # -----------------------------------------

        filename = (self.state.drawing_filename or "").strip()
        encoded_pdf = (self.state.drawing_pdf_b64 or "").strip()

        if not filename:
            raise RuntimeError(
                "drawing_filename is required."
            )

        if not filename.lower().endswith(".pdf"):
            raise RuntimeError(
                "The uploaded drawing must be a PDF file."
            )

        if not encoded_pdf:
            raise RuntimeError(
                "drawing_pdf_b64 is required."
            )

        # Support both plain Base64 and data-URL format.
        if encoded_pdf.startswith("data:"):
            try:
                encoded_pdf = encoded_pdf.split(",", 1)[1]
            except IndexError as exc:
                raise RuntimeError(
                    "Invalid PDF data URL."
                ) from exc

        # Remove whitespace/newlines that may be introduced in transport.
        encoded_pdf = re.sub(r"\s+", "", encoded_pdf)

        # -----------------------------------------
        # DECODE PDF TO TEMPORARY FILE
        # -----------------------------------------

        try:
            pdf_bytes = base64.b64decode(
                encoded_pdf,
                validate=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "drawing_pdf_b64 is not valid Base64."
            ) from exc

        if not pdf_bytes.startswith(b"%PDF"):
            raise RuntimeError(
                "Decoded content is not a valid PDF file."
            )

        temp_path = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=".pdf",
                delete=False,
            ) as temp_file:
                temp_file.write(pdf_bytes)
                temp_path = Path(temp_file.name)

            print(
                f"\nPDF received: {filename}"
            )

            # -----------------------------------------
            # READ PDF USING CUSTOM CREWAI TOOL
            # -----------------------------------------

            reader = DrawingPDFReaderTool()

            pdf_result = reader.run(
                file_path=str(temp_path)
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

            self.state._internal["pdf_text"] = pdf_result
            self.state._internal["drawing_filename"] = filename

            return pdf_result

        finally:
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass


    @listen(receive_and_read_pdf)
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
                 "Classify construction and engineering drawings accurately, "
                 "detect any mismatch between the stated discipline and the actual "
                 "drawing content, and route mismatches for human review."
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

Determine the ACTUAL discipline from the overall drawing content,
symbols, annotations, room/layout information, terminology,
drawing features and other discipline-specific evidence.

Also identify the discipline explicitly stated in the drawing title,
drawing number prefix, title block or metadata.

If the actual drawing content indicates a different discipline from
the stated discipline, report a Discipline Mismatch.

Do not automatically trust the written title if the drawing content
clearly indicates another discipline.

Example:

If the drawing content is clearly Architectural but the title says
Structural, classify the detected discipline as Architectural and return:

Detected Discipline: Architectural
Stated Discipline: Structural
Discipline Match: No
Mismatch Warning: Drawing content is Architectural, but the drawing title states Structural.
Recommended Action: Review and correct the drawing title/metadata before routing.

If the detected discipline and stated discipline agree:

Discipline Match: Yes
Mismatch Warning: None
Recommended Action: None

Confidence:

95-100 = Explicit discipline with multiple strong indicators
85-94  = Strong consistent evidence
80-84  = Clear classification with minor ambiguity
60-79  = Moderate evidence requiring human review
Below 60 = Weak or conflicting evidence

Return exactly:

Drawing Title: <title>
Drawing Number: <number>
Detected Discipline: <discipline determined from actual drawing content>
Stated Discipline: <discipline stated in drawing title, drawing number, or metadata>
Discipline Match: <Yes or No>
Confidence: <number>%
Reason: <short evidence-based reason>
Mismatch Warning: <warning if mismatch exists, otherwise None>
Recommended Action: <action required if mismatch exists, otherwise None>
""",

            expected_output=(
                "Drawing Title, Drawing Number, Detected Discipline, "
                "Stated Discipline, Discipline Match, Confidence percentage, "
                "classification reason, mismatch warning and recommended action."
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

        data = self.state._internal

        data["drawing_title"] = get_field(
            classification_result,
            "Drawing Title",
        )

        data["drawing_number"] = get_field(
            classification_result,
            "Drawing Number",
        )

        data["detected_discipline"] = get_field(
            classification_result,
            "Detected Discipline",
        )

        data["stated_discipline"] = get_field(
            classification_result,
            "Stated Discipline",
        )

        data["discipline_match"] = get_field(
            classification_result,
            "Discipline Match",
        )

        # Keep the existing discipline key for downstream routing.
        data["discipline"] = data["detected_discipline"]

        confidence_text = get_field(
            classification_result,
            "Confidence",
        )

        data["reason"] = get_field(
            classification_result,
            "Reason",
        )

        data["mismatch_warning"] = get_field(
            classification_result,
            "Mismatch Warning",
        )

        data["recommended_action"] = get_field(
            classification_result,
            "Recommended Action",
        )

        confidence_match = re.search(
            r"(\d+)",
            confidence_text,
        )

        if confidence_match:
            data["confidence"] = int(
                confidence_match.group(1)
            )
        else:
            data["confidence"] = 0

        return classification_result


    @listen(classify_drawing)
    def route_by_confidence(self):

        data = self.state._internal

        # -----------------------------------------
        # ROUTING RULE
        # -----------------------------------------

        discipline_match = (
            str(data.get("discipline_match", "No"))
            .strip()
            .lower()
        )

        mismatch_exists = discipline_match in {
            "no",
            "false",
            "mismatch",
        }

        if mismatch_exists:

            data["routing_decision"] = (
                "Send for Human Review - Discipline Mismatch"
            )

            data["human_review_required"] = "Yes"
            data["reviewer_name"] = "Pending"
            data["final_discipline"] = "Pending Human Review"
            data["final_action"] = (
                "Awaiting Human Review - Discipline Mismatch"
            )

            print(
                "\n================ DISCIPLINE MISMATCH ================\n"
            )

            print(
                f"Detected Discipline: "
                f"{data.get('detected_discipline', 'Not Found')}"
            )

            print(
                f"Stated Discipline: "
                f"{data.get('stated_discipline', 'Not Found')}"
            )

            print(
                f"Mismatch Warning: "
                f"{data.get('mismatch_warning', 'Not Found')}"
            )

            print(
                f"Recommended Action: "
                f"{data.get('recommended_action', 'Not Found')}"
            )

        elif data.get("confidence", 0) >= 80:

            data["routing_decision"] = (
                f"Auto-route to "
                f"{data.get('discipline', 'Other')} workflow"
            )

            data["human_review_required"] = "No"
            data["reviewer_name"] = "N/A"
            data["final_discipline"] = data.get(
                "discipline",
                "Other",
            )
            data["final_action"] = (
                "Automatically Routed"
            )

        else:

            data["routing_decision"] = (
                "Send for Human Review"
            )

            data["human_review_required"] = "Yes"
            data["reviewer_name"] = "Pending"
            data["final_discipline"] = "Pending Human Review"
            data["final_action"] = "Awaiting Human Review"

            print(
                "\n================ HUMAN REVIEW REQUIRED ================\n"
            )

            print(
                f"AI Classification: "
                f"{data.get('discipline', 'Not Found')}"
            )

            print(
                f"Confidence: "
                f"{data.get('confidence', 0)}%"
            )

            print(
                f"Reason: "
                f"{data.get('reason', 'Not Found')}"
            )

        return data["routing_decision"]


    @listen(route_by_confidence)
    def create_audit_record(self):

        data = self.state._internal

        # -----------------------------------------
        # CREATE FINAL AUDIT RECORD
        # -----------------------------------------

        date_time = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        audit_record = f"""
================ FINAL AUDIT RECORD ================

Drawing Title: {data.get("drawing_title", "Not Found")}
Drawing Number: {data.get("drawing_number", "Not Found")}
Detected Discipline: {data.get("detected_discipline", "Not Found")}
Stated Discipline: {data.get("stated_discipline", "Not Found")}
Discipline Match: {data.get("discipline_match", "Not Found")}
Confidence: {data.get("confidence", 0)}%
Classification Reason: {data.get("reason", "Not Found")}
Mismatch Warning: {data.get("mismatch_warning", "Not Found")}
Recommended Action: {data.get("recommended_action", "Not Found")}
Routing Decision: {data.get("routing_decision", "Not Found")}
Human Review Required: {data.get("human_review_required", "Not Found")}
Reviewer Name: {data.get("reviewer_name", "Not Found")}
Final Discipline: {data.get("final_discipline", "Not Found")}
Date/Time: {date_time}
Final Action: {data.get("final_action", "Not Found")}

====================================================
"""

        data["audit_record"] = audit_record

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


if __name__ == "__main__":
    kickoff()
