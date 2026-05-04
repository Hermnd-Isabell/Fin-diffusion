
try:
    with open('analysis_output.txt', 'r', encoding='utf-16le') as f:
        content = f.read()
    
    with open('analysis_output_utf8.txt', 'w', encoding='utf-8') as f:
        f.write(content)
        
    print("Conversion successful.")
except Exception as e:
    print(f"Error converting file: {e}")
