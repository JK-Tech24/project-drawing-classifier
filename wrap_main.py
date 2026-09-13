from pathlib import Path

p = Path("src/project_drawing_classifier/main.py")
s = p.read_text(encoding="utf-8")

marker = "# -------------------------------------------------\n# LOAD GEMINI API KEY"

head, body = s.split(marker, 1)
body = marker + body

indented = "\n".join(
    "    " + line if line else ""
    for line in body.splitlines()
)

new_text = (
    head.rstrip()
    + "\n\n\ndef main():\n"
    + indented
    + "\n"
)

p.write_text(new_text, encoding="utf-8")

print("main.py wrapped successfully")
