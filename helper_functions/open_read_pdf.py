import fitz
from helper_functions.text_formatter import text_formatter


def open_read_pdf(pdf_path: str) -> list[dict]:
    
    pdf_document = fitz.open(pdf_path)
    pages_and_text = []
    for page_number, page in enumerate(pdf_document.pages()):
        text = page.get_text()
        text = text_formatter(text)
        pages_and_text.append({"page_number": page_number,  # adjust page numbers since our PDF starts on page 42
                                "page_char_count": len(text),
                                "page_word_count": len(text.split(" ")),
                                "page_sentence_count_raw": len(text.split(". ")),
                                "page_token_count": len(text) / 4,  # 1 token = ~4 chars, see: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
                                "text": text})
    return pages_and_text
