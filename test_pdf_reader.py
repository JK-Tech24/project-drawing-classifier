from project_drawing_classifier.tools.drawing_pdf_reader import DrawingPDFReaderTool

tool = DrawingPDFReaderTool()

result = tool.run(
   file_path=r"C:\Users\JeffKomati\Downloads\Page_3_Test_Drawing.pdf"
)

print(result)