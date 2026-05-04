
try:
    with open('analysis_output.txt', 'r', encoding='utf-16le') as f:
        print(f.read())
except Exception as e:
    print(f"Error reading file with utf-16le: {e}")
    try:
        with open('analysis_output.txt', 'r', encoding='utf-8') as f:
            print(f.read())
    except Exception as e2:
        print(f"Error reading file with utf-8: {e2}")
