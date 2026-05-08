import zipfile
import xml.etree.ElementTree as ET
import sys

def read_docx(path):
    try:
        document = zipfile.ZipFile(path)
        xml_content = document.read('word/document.xml')
        document.close()
        tree = ET.XML(xml_content)
        
        WORD_NAMESPACE = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
        PARA = WORD_NAMESPACE + 'p'
        TEXT = WORD_NAMESPACE + 't'
        
        paragraphs = []
        for paragraph in tree.iter(PARA):
            texts = [node.text for node in paragraph.iter(TEXT) if node.text]
            if texts:
                paragraphs.append(''.join(texts))
                
        return '\n'.join(paragraphs)
    except Exception as e:
        return f"Error reading docx: {e}"

def read_pdf(path):
    try:
        import PyPDF2
        with open(path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = []
            for i in range(min(5, len(reader.pages))): # read first 5 pages
                text.append(reader.pages[i].extract_text())
            return '\n'.join(text)
    except ImportError:
        try:
            import fitz
            doc = fitz.open(path)
            text = []
            for i in range(min(5, doc.page_count)):
                text.append(doc[i].get_text())
            return '\n'.join(text)
        except ImportError:
            return "Neither PyPDF2 nor PyMuPDF (fitz) is installed."
    except Exception as e:
        return f"Error reading pdf: {e}"

if __name__ == '__main__':
    docx_path = r"c:\Users\gauta\K-means_clustering\data\MLReport (1).docx"
    pdf_path = r"c:\Users\gauta\K-means_clustering\data\Unboxing_K-Means.pdf"
    
    print("=== DOCX START ===")
    docx_text = read_docx(docx_path)
    print(docx_text[:5000]) # Print first 5000 chars to avoid overwhelming output
    print("=== DOCX END ===")
    
    print("\n=== PDF START ===")
    print(read_pdf(pdf_path))
    print("=== PDF END ===")
