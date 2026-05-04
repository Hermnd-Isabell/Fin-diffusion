import pypdf

pdf_path = "扩散模型生成期权仿真数据.pdf"
output_path = "pdf_content_utf8.txt"

try:
    reader = pypdf.PdfReader(pdf_path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Total pages: {len(reader.pages)}\n\n")
        
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            f.write(f"--- Page {i+1} ---\n")
            f.write(text)
            f.write("\n\n")
            
    print(f"Successfully wrote content to {output_path}")
        
except Exception as e:
    print(f"Error reading PDF: {e}")
