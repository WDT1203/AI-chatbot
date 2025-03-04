import time
import random
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# Fetch and parse the sitemap.xml (list all allowed URLs)
def fetch_sitemap(url):
    retries = 3  # Number of retry attempts
    while retries > 0:
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raises HTTPError for bad responses
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'xml')
                urls = [loc.text for loc in soup.find_all('loc')]
                return urls
            else:
                print(f"Failed to fetch sitemap. HTTP Status: {response.status_code}")
                return []
        except requests.exceptions.RequestException as e:
            print(f"Error fetching sitemap: {e}")
            retries -= 1
            if retries > 0:
                print(f"Retrying... ({3 - retries} out of 3)")
                time.sleep(5)
            else:
                print("Max retries exceeded.")
                return []

# Load webpage content (consider all sub-pages and links)
def load_webpages(urls):
    loader = WebBaseLoader(urls)
    documents = loader.load()
    return documents

# Crawl sub-pages by following links from the main page
def scrape_page_and_links(url, visited=set(), max_retries=3):
    if url in visited:
        return []
    visited.add(url)
    
    retries = 0
    while retries < max_retries:
        try:
            # Make request to the URL with a delay to avoid being blocked
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            if response.status_code != 200:
                if response.status_code == 429:  # Too Many Requests
                    print(f"Rate limit hit, sleeping for 60 seconds before retrying {url}...")
                    time.sleep(60)  # Sleep for 60 seconds before retrying
                else:
                    print(f"Failed to retrieve {url}: Status code {response.status_code}")
                    return []
            else:
                break
        except requests.exceptions.RequestException as e:
            print(f"Error retrieving {url}: {e}")
            retries += 1
            time.sleep(5)  # Wait before retrying
    
    # Parse the page
    soup = BeautifulSoup(response.content, 'html.parser')
    content = soup.get_text()

    # Find all links in the page
    links = soup.find_all('a', href=True)
    sub_urls = []
    for link in links:
        href = link['href']
        full_url = urljoin(url, href)  # Get the absolute URL
        sub_urls.append(full_url)
        print(f"Found sub-url: {full_url}")


    # Random delay between requests to simulate human behavior
    time.sleep(random.uniform(1, 3))  # Sleep between 1 and 3 seconds before making the next request
    
    # Recursively scrape sub-pages
    sub_documents = []
    for sub_url in sub_urls:
        sub_documents += scrape_page_and_links(sub_url, visited)

    return [content] + sub_documents

# List of websites you want to scrape
base_url = "https://nida.nih.gov/"
sitemap_url = "https://nida.nih.gov/sitemap.xml"

# Fetch sitemap URLs
sitemap_urls = fetch_sitemap(sitemap_url)

# Start scraping
all_documents = load_webpages(urls=sitemap_urls)

# Create chunks of text for embedding
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(extracted_data=all_documents)

# Get embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Path to FAISS database
DB_FAISS_PATH = "vectorstore/db_faiss"

# Load existing FAISS database if available, or create a new one
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

print(f"Saved FAISS database at {DB_FAISS_PATH}")
