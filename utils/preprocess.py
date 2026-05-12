import re
import unicodedata
from typing import List
from underthesea import word_tokenize

LEGAL_STOPWORDS = set([
    "là", "và", "của", "các", "có", "được", "những", "sự",
    "cho", "trong", "tại", "theo", "về", "việc", "này", "kia", "đó", "đây",
    "thì", "mà", "rằng", "hoặc", "hay", "như", "nhưng", "tuy", "nên",
    "bằng", "để", "do", "bởi", "với", "cùng",
    "từ", "đến", "khi", "sau", "nay", "trước", "đang", "đã", "sẽ",
    "bị", "lại", "ra", "vào", "lên", "xuống", "qua", "vừa", "mới",
    "nào", "gì", "ai", "sao", "vậy", "bao_nhiêu", "thế_nào", "bao_lâu",
    "đối_với", "quy_định", "căn_cứ", "áp_dụng", "thực_hiện", "ban_hành",
    "thi_hành", "hiệu_lực", "bao_gồm", "gồm", "thuộc", "cũng", "nêu",
    "quy_định_tại", "trường_hợp", "cụ_thể", "liên_quan"
])

def clean_text(text: str) -> List[str]:

    if not text or not isinstance(text, str):
        return []

    text = unicodedata.normalize('NFC', text)
    text = text.lower()
    
    text = re.sub(r'[^\w\s/.-]', ' ', text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return []

    segmented_text = word_tokenize(text, format="text")
    tokens = segmented_text.split()
    valid_tokens = [t for t in tokens if t not in ['/', '-', '.']]
    final_tokens = [t for t in valid_tokens if t not in LEGAL_STOPWORDS]
    
    return final_tokens