import streamlit as st
import os
from rag_backend import initialize_rag_system

st.set_page_config(page_title="Local RAG System", page_icon="🤖", layout="centered")
st.title("🤖 Enterprise Local RAG System")
st.write("An air-gapped, private Q&A system running entirely on your local machine via LM Studio.")

# Cache the data processing so it doesn't re-run every time you click a button
@st.cache_resource
def load_rag():
    if not os.path.exists("./document.pdf"):
        return None
    return initialize_rag_system("./document.pdf")

# Check if the document exists before running the app setup
if not os.path.exists("./document.pdf"):
    st.info("💡 Place a PDF file named `document.pdf` into this project directory and refresh this page.")
else:
    with st.spinner("Processing document and initializing vector database..."):
        rag_chain = load_rag()
    
    user_query = st.text_input("Ask a question about your document:")
    
    if user_query:
        with st.spinner("Searching local vector database and generating response..."):
            try:
                response = rag_chain.invoke({"input": user_query})
                
                st.subheader("Response:")
                st.write(response["answer"])
                
                # Show source context to prove it isn't hallucinating
                with st.expander("📚 View Document Sources Used"):
                    for i, doc in enumerate(response["context"]):
                        page_num = doc.metadata.get('page', i + 1)
                        st.markdown(f"**Source Chunk {i+1} (Page {page_num}):**")
                        st.caption(doc.page_content)
                        st.write("---")
            except Exception as e:
                st.error("Error connecting to LM Studio. Make sure your local server is running on port 1234!")