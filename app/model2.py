import re
import cv2
import numpy as np
from pdf2image import convert_from_path
import easyocr
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus.common import thai_words, thai_stopwords
from pythainlp.spell import spell
from pythainlp.tokenize import sent_tokenize

class Model2:
    def __init__(self, data=None, langs=('th','en')):
        self.data = data
        self.ocr_reader = easyocr.Reader(langs, gpu=False)
        self.dict_words = set(thai_words())
        self.stop_words = set(thai_stopwords())

        self.re_num   = re.compile(r"^\d+([.,:/-]\d+)*$")
        self.re_eng   = re.compile(r"^[A-Za-z0-9_+\-./]+$")

    @staticmethod
    def joinText(ocr_result, sep=" ") -> str:
        texts = [text.strip() for (_, text, _) in ocr_result if text and text.strip()]
        return sep.join(texts)
    
    def preprocessOCR(self, img_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th 

    def extractFields(self, text: str) -> dict:
        pattern = {
            'หัวข้อ': r'(?:หัวข้อ(?:ปัญหาพิเศษ|สหกิจศึกษา|โครงงานพิเศษ)|สหกิจศึกษา)\s*(.*?)(?=\sชื่อนักศึกษา|$)',
            'ชื่อนักศึกษา': r'ชื่อนักศึกษา\s*(.*?)(?=\sปริญญา|$)',
            'ปริญญา': r'ปริญญา\s*(.*?)(?=\sภาควิชา|$)',
            'ภาควิชา': r'ภาควิชา\s*(.*?)(?=\sคณะ|ปีการศึกษา|$)',
            'คณะ': r'คณะ\s*(.*?)(?=\sมหาวิทยาลัย|$)',
            'มหาวิทยาลัย': r'มหาวิทยาลัย\s*(.*?)(?=\sปีการศึกษา|$)',
            'ปีการศึกษา': r'ปีการศึกษา\s*(.*?)(?=\sอาจารย์ที่ปรึกษา|$)',
            'อาจารย์ที่ปรึกษา': r'อาจารย์ที่ปรึกษา\s*(.*?)(?=\sบทคัดย่อ|$)',
            'บทคัดย่อ': r'บทคัดย่อ\s*(.*?)(?=\sคำสำคัญ|$)',
            'คำสำคัญ': r'(?:คำสำคัญ:|คำสำคัญ)\s*(.*?)(?=\sTitle|$)',
        }
        results = {}
        for key, pat in pattern.items():
            m = re.search(pat, text, flags=re.DOTALL)
            if m:
                results[key] = m.group(1).strip()
        return results
    
    def shouldSkip(self, tok: str) -> bool:
        t = tok.strip()
        if not t:
            return True
        if self.re_num.match(t) or self.re_eng.match(t):
            return True
        return False
    
    def check_tokens(self, text: str):
        tokens = word_tokenize(text or "", engine="newmm")
        results = []

        for tok in tokens:
            if self.shouldSkip(tok):
                results.append((tok, 'ok', []))
                continue

            if len(tok) <= 2:
                if tok in self.dict_words or tok in self.stop_words:
                    results.append((tok, 'ok', []))
                else:
                    results.append((tok, 'unknown', []))
                continue

            if tok in self.dict_words or tok in self.stop_words:
                results.append((tok, 'ok', []))
            else:
                suggestions = spell(tok) or []
                results.append((tok, 'error', suggestions[:5]))

        return results

    def pdfToImage(self, file_path: str, page_num: int = 1, poppler_path: str = None) -> np.ndarray:
        pages = convert_from_path(
            file_path,
            dpi=300,
            first_page=page_num,
            last_page=page_num,
            poppler_path=poppler_path  
        )
        pil_img = pages[0]                
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)  
        return cv_img
    
    def processDocumentOCR(self, file_path: str, page_num: int = 4,poppler_path: str = None) -> dict:
        img_bgr = self.pdfToImage(file_path, page_num=page_num, poppler_path=poppler_path)
        img_bin =  self.preprocessOCR(img_bgr)
        img_rgb = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2RGB)

        ocr_result = self.ocr_reader.readtext(img_rgb)
        sentence = self.joinText(ocr_result, sep=" ")
        fields = self.extractFields(sentence)
        #report = self.check_tokens(sentence)
        return fields
    
    def check_fields(self, fields: dict):
        report = {}
        for fname, text in (fields or {}).items():
            token_report = self.check_tokens(text)
            errors = [(tok, sugg) for tok, status, sugg in token_report if status == "error"]
            unknowns = [tok for tok, status, _ in token_report if status == "unknown"]

            if errors:
                status = "error"
            elif unknowns:
                status = "mixed"
            else:
                status = "ok"

            report[fname] = {
                "status": status,
                "errors": errors,
                "unknowns": unknowns
            }
        return report
