import time
import streamlit as st
from rag_engine import collection,retrieve,generate_answer
st.set_page_config(page_title='BudgetLens RAG',page_icon='📚',layout='wide')
st.title('📚 BudgetLens — Explainable PDF RAG')
st.caption('Open-source embeddings + Chroma vector search | Evidence-grounded US Federal Budget QA')
try: count=collection().count()
except Exception as ex: st.error(f'Database error: {ex}');st.stop()
a,b,c=st.columns(3)
a.metric('Indexed chunks',f'{count:,}')
b.metric('Corpus', '10 PDF files (2015–2024)')
c.metric('Mode','Local embeddings')
st.info('Index coverage is shown by indexed chunks. Full-corpus ingestion must be run separately. Answers are extractive until an LLM API is configured.')
with st.sidebar:
    st.header('Retrieval controls')
    topk=st.slider('Top-K passages',2,10,5)
    st.markdown('**Pipeline**\n\nPDF → page extraction → overlapping chunks → MiniLM embeddings → Chroma ANN → grounded response')
    st.markdown('**Example questions**')
    st.code('What does the 2020 budget say about the Department of Agriculture?')
    st.code('Which agencies are discussed in the 2018 appendix?')
    st.caption('Evaluation: track indexed coverage, per-query latency and evidence citations. Do not present unmeasured p95 or accuracy scores.')
if 'history' not in st.session_state:st.session_state.history=[]
for question,answer in st.session_state.history:
    with st.chat_message('user'):st.write(question)
    with st.chat_message('assistant'):st.markdown(answer)
question=st.chat_input('Ask a question about the indexed budget PDFs')
if question:
    with st.chat_message('user'):st.write(question)
    with st.chat_message('assistant'):
        if not count:st.warning('No documents indexed. Run: python ingest.py --pages-per-pdf 30')
        else:
            start=time.perf_counter()
            with st.spinner('Retrieving relevant evidence...'):
                chunks,retrieval_sec=retrieve(question,topk)
                answer,mode=generate_answer(question,chunks)
            elapsed=time.perf_counter()-start
            st.markdown(answer)
            st.caption(f'Mode: {mode} | Retrieval: {retrieval_sec:.2f}s | End-to-end: {elapsed:.2f}s | Retrieved: {len(chunks)}')
            with st.expander('🔎 Retrieved evidence and similarity scores',expanded=True):
                for i,ch in enumerate(chunks,1):
                    st.markdown(f'**{i}. {ch["source"]} · page {ch["page"]} · score {ch["score"]:.3f}**')
                    st.text(ch['text'][:800])
            st.session_state.history.append((question,answer))
