import hashlib

import chromadb
import ollama
import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings


class PDFChatbot:
    def __init__(self, model_name="deepseek-r1"):
        self.model_name = model_name
        self.embeddings = OllamaEmbeddings(model=model_name)
        self.chroma_client = chromadb.PersistentClient(path="./chroma_storage")
        self.collection = None

    def _get_hash_for_pdf(self, pdf_file):
        return hashlib.md5(pdf_file.read()).hexdigest()

    def process_pdf(self, pdf_file):
        pdf_reader = pypdf.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100)
        texts = text_splitter.split_text(text)

        self.collection = self.chroma_client.create_collection(
            name=self._get_hash_for_pdf(pdf_file), get_or_create=True)
        for i, chunk in enumerate(texts):
            self.collection.add(
                embeddings=self.embeddings.embed_query(chunk),
                documents=[chunk],
                ids=[f"chunk_{i}"]
            )

    def query_pdf(self, query):
        if not self.collection:
            return "No PDF processed yet."

        query_embedding = self.embeddings.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )

        context = " ".join(results['documents'][0])

        full_prompt = f"""
        Context: The user has uploaded a PDF file containing the following text: {context}
        
        Question: {query}
        
        Answer based on the context:
        """
        response = ollama.chat(model=self.model_name, messages=[
            {'role': 'user', 'content': full_prompt}
        ])

        return response['message']['content']
