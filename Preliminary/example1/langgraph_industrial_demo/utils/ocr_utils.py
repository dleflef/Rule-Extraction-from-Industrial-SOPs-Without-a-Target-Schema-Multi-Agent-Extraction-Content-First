import pdfplumber
import pytesseract
from pdf2image import convert_from_path
import os

def extract_text_from_pdf(pdf_path: str) -> str:
    # Fallback OCR strategy: Direct extraction -> Tesseract OCR.
    if not pdf_path or not os.path.exists(pdf_path):
        return f"Error: File '{pdf_path}' not found or path is empty."

    extracted_text = ""
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Loop through each page and extract text
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"
    except Exception as e:
        print(f"Direct extraction failed: {e}")

    # Fallback to OCR if text is sparse
    if len(extracted_text.strip()) < 100:
        try:
            # Convert PDF pages to images and then apply OCR 
            images = convert_from_path(pdf_path)
            for image in images:
                text = pytesseract.image_to_string(image)
                extracted_text += text + "\n"
        except Exception as e:

            extracted_text = f"[OCR Failed. Ensure Poppler and Tesseract are installed.] Error: {e}"
    
    # Sends extracted text back to the agent for processing
    return extracted_text
