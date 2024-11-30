from lxml import etree
import os

def extract_text(element):
    text = []
    for el in element.iter():
        if el.tag.endswith("p") and el.text:
            text.append(el.text.strip())
    return "\n".join(text)

def parse_fb2(file_path):
    tree = etree.parse(file_path)
    namespace = {"ns": "http://www.gribuser.ru/xml/fictionbook/2.0"}
    body = tree.find(".//ns:body", namespaces=namespace)
    if body is None:
        return "No body tag found in FB2 file."
    return extract_text(body)

def process_books(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    files = [f for f in os.listdir(input_dir) if f.endswith('.fb2')]
    for i, file in enumerate(files, start=1):
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, f"book_{i}.txt")
        text = parse_fb2(input_path)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Processed {file} -> {output_path}")

if __name__ == "__main__":
    raw_data_dir = "./data/raw/"
    processed_data_dir = "./data/processed/"
    process_books(raw_data_dir, processed_data_dir)
