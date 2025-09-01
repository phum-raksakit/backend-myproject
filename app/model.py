from csv import reader
from pydoc import doc
import re
import fitz

class Model:
    def __init__(self, data=None):
        self.data = data

    def getText(self, page):
        text = page.get_text("text")
        return text
    
    def cleanText(self, text):
        clean_text = " ".join(text.split())
        return clean_text

    def extractFields(self, text):
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
    
    def processDocument(self, doc: fitz.Document):
        page = doc[3]
        text = self.getText(page)
        print(text)
        cleaned = self.cleanText(text)
        fields = self.extractFields(cleaned)
        return fields
