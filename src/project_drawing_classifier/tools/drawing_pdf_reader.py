from pathlib import Path
from typing import Type

import fitz
from pydantic import BaseModel, Field
from crewai.tools import BaseTool


class DrawingPDFReaderInput(BaseModel):
    file_path: str = Field(
        ...,
        description="Full local path to the PDF drawing file."
    )


class DrawingPDFReaderTool(BaseTool):
    name: str = "Drawing PDF Reader"
    description: str = (
        "Reads text from a construction or engineering PDF drawing "
        "using the local file path."
    )

    args_schema: Type[BaseModel] = DrawingPDFReaderInput

    def _run(self, file_path: str) -> str:
        try:
            path = Path(file_path).expanduser().resolve()

            if not path.exists():
                return (
                    "PDF_READ_FAILED\n"
                    f"Reason: File does not exist.\n"
                    f"Resolved path: {path}"
                )

            if not path.is_file():
                return (
                    "PDF_READ_FAILED\n"
                    f"Reason: Path is not a file.\n"
                    f"Resolved path: {path}"
                )

            if path.suffix.lower() != ".pdf":
                return (
                    "PDF_READ_FAILED\n"
                    "Reason: File is not a PDF."
                )

            document = fitz.open(str(path))

            pages = []
            total_characters = 0

            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text").strip()

                if text:
                    total_characters += len(text)
                    pages.append(
                        f"\n===== PAGE {page_number} =====\n{text}"
                    )

            page_count = len(document)
            document.close()

            if total_characters == 0:
                return (
                    "PDF_TEXT_EMPTY\n"
                    f"PAGE_COUNT: {page_count}\n"
                    "The PDF opened successfully but no embedded text was found."
                )

            extracted_text = "\n".join(pages)

            return (
                "PDF_READ_SUCCESS\n"
                f"FILE_NAME: {path.name}\n"
                f"PAGE_COUNT: {page_count}\n"
                f"TEXT_CHAR_COUNT: {total_characters}\n"
                "\n--- BEGIN PDF TEXT ---\n"
                f"{extracted_text}\n"
                "--- END PDF TEXT ---"
            )

        except Exception as exc:
            return (
                "PDF_READ_FAILED\n"
                f"Reason: {type(exc).__name__}: {exc}"
            )