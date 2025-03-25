from pypdf import PdfReader

import os

if __name__ == "__main__":
    pdf_path = "RawDocs/Computer Architecture-6th.pdf"
    fd = open(pdf_path, "rb")
    pdf = PdfReader(fd)
    print(pdf.get_page(1142), "\n========================\n")
    print(pdf.pages[1142].extract_text())
    fd.close()

