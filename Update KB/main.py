import requests

from bs4 import BeautifulSoup

import pandas as pd

from process_record import process_record, added_records, updated_records
from compare_record import compare_records
from delete_record import delete_record, deleted_records


def fetch_data():
    data = []

    for i in range(1, 84):
        url = f"https://peraturan.go.id/cari?PeraturanSearch%5Btentang%5D=&PeraturanSearch%5Bnomor%5D=&PeraturanSearch%5Btahun%5D=&PeraturanSearch%5Bjenis_peraturan_id%5D=3&PeraturanSearch%5Bpemrakarsa_id%5D=&PeraturanSearch%5Bstatus%5D=Berlaku&page={i}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        undang_undang_elements = []
        
        for element in soup.find_all(string=lambda string: string and string.strip().startswith("Undang-Undang Nomor")):
            undang_undang_elements.append(element.strip())
            parent = element.parent
            sibling = parent.find_next_sibling()
            
            if sibling:
                description = sibling.get_text().strip()
                data.append({
                    "nama": element.strip(),
                    "description": description
                })
            
            document_html = parent.parent.find_next_sibling()
            link_elements = document_html.find_all('a')
            file_names = []
            real_file_names = []
            for link_element in link_elements:
                if link_element.has_attr('href'):
                    href = link_element['href']
                    file_names.append(href.split("/files/")[1].replace("-", "_").lower().replace("+", "_").replace("%", "_").replace("__", "_").replace("..", ".").replace("._", "_"))
                    real_file_names.append(href)
            data[-1]["file_name"] = file_names
            data[-1]["real_file_name"] = real_file_names

    df = pd.DataFrame(data)
    df.to_csv("updated_data.csv", index=False)

def update_records():
    fetch_data()
    compare_records()
    delete_record(deleted_records)
    process_record(added_records)
    process_record(updated_records)

if __name__ == "__main__":
    update_records()