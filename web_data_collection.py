from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Path to FAISS database
DB_FAISS_PATH = "vectorstore/db_faiss"

# Load webpage content
def load_webpages(urls):
    loader = WebBaseLoader(urls)
    documents = loader.load()
    return documents

# List of authoritative URLs
urls = [
    "https://nida.nih.gov/publications/drugfacts/understanding-drug-use-addiction",
    "https://nida.nih.gov/research-topics/co-occurring-disorders-health-conditions",
    "https://nida.nih.gov/research-topics/treatment",
    "https://nida.nih.gov/research-topics/covid-19-substance-use",
    "https://nida.nih.gov/research-topics/drug-checking",
    "https://nida.nih.gov/research-topics/drug-testing",
    "https://nida.nih.gov/publications/drugfacts/drugged-driving",
    "https://nida.nih.gov/research-topics/drugs-brain",
    "https://nida.nih.gov/research-topics/trends-statistics/overdose-death-rates",
    "https://nida.nih.gov/research-topics/harm-reduction",
    "https://nida.nih.gov/research-topics/viral-hepatitis",
    "https://nida.nih.gov/research-topics/hiv",
    "https://nida.nih.gov/research-topics/trends-statistics/infographics/reported-reasons-vaping-among-us-adolescents-2021-2023",
    "https://nida.nih.gov/research-topics/opioids/medications-opioid-overdose-withdrawal-addiction-infographic",
    "https://nida.nih.gov/research-topics/college-age-young-adults/vaping-cannabis-trends-among-young-adults-infographic",
    "https://nida.nih.gov/research-topics/college-age-young-adults/drug-alcohol-use-in-college-age-adults-2017-infographic",
    "https://nida.nih.gov/research-topics/mental-health",
    "https://nida.nih.gov/research-topics/trends-statistics/monitoring-future",
    "https://nida.nih.gov/research-topics/trends-statistics/national-drug-early-warning-system-ndews",
    "https://nida.nih.gov/research-topics/overdose-prevention-centers",
    "https://nida.nih.gov/research-topics/overdose-reversal-medications",
    "https://nida.nih.gov/research-topics/recovery",
    "https://nida.nih.gov/research-topics/stigma-discrimination",
    "https://nida.nih.gov/research-topics/syringe-services-programs",
    "https://nida.nih.gov/research-topics/trauma-and-stress",
    "https://nida.nih.gov/research-topics/treatment",
    "https://nida.nih.gov/research-topics/trends-statistics",
    "https://nida.nih.gov/research-topics/addiction-science/words-matter-preferred-language-talking-about-addiction",

    # "https://www.samhsa.gov/find-help/prevention",
    # "https://www.cdc.gov/drugoverdose/featured-topics/treatment-recovery.html",
    # Add more URLs as needed
]

documents = load_webpages(urls=urls)
print("Number of documents loaded:", len(documents))

# Create chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(extracted_data=documents)
print("Number of text chunks:", len(text_chunks))

# Get embedding model
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model = get_embedding_model()

# Load existing FAISS database (if available) and add new data
if os.path.exists(DB_FAISS_PATH) and os.listdir(DB_FAISS_PATH):
    try:
        # Allow dangerous deserialization for trusted sources
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        print("Existing FAISS database loaded. Adding new data...")
        db.add_documents(text_chunks)
    except Exception as e:
        print(f"Error loading FAISS database: {e}. Creating a new one.")
        db = FAISS.from_documents(text_chunks, embedding_model)
else:
    print("No existing FAISS database found or directory is empty. Creating a new one.")
    db = FAISS.from_documents(text_chunks, embedding_model)
# Save updated FAISS database
db.save_local(DB_FAISS_PATH)
print("Updated FAISS database saved successfully.")
print(f"Total number of documents in FAISS: {db.index.ntotal}")