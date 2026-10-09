import concurrent.futures
import json
from pathlib import Path

import fitz
import requests

DEST = Path("data/pdfs")
DEST.mkdir(parents=True, exist_ok=True)

YEARS = range(2015, 2025)

def download(year):
    url = (
        f"https://www.govinfo.gov/content/pkg/"
        f"BUDGET-{year}-APP/pdf/BUDGET-{year}-APP.pdf"
    )
    path = DEST / f"US_Federal_Budget_{year}.pdf"

    try:
        if not path.exists():
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            if not response.content.startswith(b"%PDF"):
                raise ValueError("Downloaded content is not a PDF")
            path.write_bytes(response.content)

        with fitz.open(path) as doc:
            pages = len(doc)

        if pages < 200:
            raise ValueError(f"Only {pages} pages")

        return {
            "year": year,
            "filename": path.name,
            "pages": pages,
            "source": url,
            "status": "OK"
        }

    except Exception as error:
        return {
            "year": year,
            "status": "FAILED",
            "error": str(error)
        }

if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(download, YEARS))

    results.sort(key=lambda x: x["year"])

    for result in results:
        print(result, flush=True)

    Path("data/corpus_manifest.json").write_text(
        json.dumps(results, indent=2)
    )

    valid = [r for r in results if r["status"] == "OK"]

    print("\nDATASET SUMMARY")
    print("Valid PDFs:", len(valid))
    print("Total pages:", sum(r["pages"] for r in valid))
    print("Requirement met:", len(valid) >= 10)
