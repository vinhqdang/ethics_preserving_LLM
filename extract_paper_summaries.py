import fitz # PyMuPDF
import sys
import io

def dump_paper_sections(pdf_path, out_path, num_pages=10):
    doc = fitz.open(pdf_path)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Analysis of {pdf_path}\n\n")
        f.write(f"Total Pages: {len(doc)}\n\n")
        for i in range(min(num_pages, len(doc))):
            f.write(f"## Page {i+1}\n\n")
            f.write(doc[i].get_text())
            f.write("\n\n---\n\n")
    print(f"Dumped {num_pages} pages to {out_path}")

if __name__ == "__main__":
    dump_paper_sections("2406.15513v3.pdf", "extracted_2406.md", 12)
    dump_paper_sections("2503.17682v2.pdf", "extracted_2503.md", 12)
