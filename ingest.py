import argparse, hashlib, time
from pathlib import Path
import pymupdf
from rag_engine import collection,model,chunk_text,ROOT

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pages-per-pdf',type=int,default=30,help='Pages per PDF this pass. 0 = all pages')
    p.add_argument('--batch',type=int,default=48)
    p.add_argument('--ocr',action='store_true',help='Try Tesseract OCR for pages with little native text')
    args=p.parse_args()
    pdfs=sorted((ROOT/'data'/'pdfs').glob('*.pdf'))
    if not pdfs:raise SystemExit('No PDFs in data/pdfs')
    col=collection();encoder=model();start=time.perf_counter();page_count=0;chunks_added=0;ocr_pages=0
    ids=[];texts=[];metas=[]
    def flush():
        nonlocal chunks_added
        if not ids:return
        # upsert permits resuming without duplicate records
        embeddings=encoder.encode(texts,batch_size=24,normalize_embeddings=True,show_progress_bar=False).tolist()
        col.upsert(ids=list(ids),documents=list(texts),metadatas=list(metas),embeddings=embeddings)
        chunks_added+=len(ids);ids.clear();texts.clear();metas.clear()
        print(f'  Indexed this pass: {chunks_added} chunks; DB count: {col.count()}',flush=True)
    for path in pdfs:
        with pymupdf.open(path) as pdf:
            limit=len(pdf) if args.pages_per_pdf==0 else min(args.pages_per_pdf,len(pdf))
            print(f'Processing {path.name} pages 1-{limit}/{len(pdf)}',flush=True)
            for idx in range(limit):
                page=pdf[idx];txt=page.get_text('text')
                if len(txt.strip())<80 and args.ocr:
                    try:
                        import pytesseract
                        from PIL import Image
                        pix=page.get_pixmap(matrix=pymupdf.Matrix(1.5,1.5))
                        txt=pytesseract.image_to_string(Image.frombytes('RGB' if pix.n==3 else 'L',[pix.width,pix.height],pix.samples))
                        ocr_pages+=1
                    except Exception as exc:
                        print('OCR unavailable:',str(exc)[:120],flush=True)
                page_count+=1
                for j,chunk in enumerate(chunk_text(txt)):
                    identifier=hashlib.sha256(f'{path.name}:{idx+1}:{j}:{chunk}'.encode()).hexdigest()
                    ids.append(identifier);texts.append(chunk)
                    metas.append({'filename':path.name,'page':idx+1,'year':path.stem[-4:],'pdf_id':path.stem})
                    if len(ids)>=args.batch:flush()
        flush()
    print(f'DONE pages read={page_count}, OCR pages={ocr_pages}, upserted chunks={chunks_added}, persisted total={col.count()}, seconds={time.perf_counter()-start:.1f}',flush=True)
if __name__=='__main__':main()
