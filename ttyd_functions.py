
import datetime
import uuid
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
from langchain.document_loaders import WebBaseLoader, TextLoader, Docx2txtLoader, PyMuPDFLoader
from whatsapp_chat_custom import WhatsAppChatLoader # use this instead of from langchain.document_loaders import WhatsAppChatLoader

from collections import deque
import re
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse
import mimetypes
from pathlib import Path
import tiktoken

# Regex pattern to match a URL
HTTP_URL_PATTERN = r'^http[s]*://.+'

mimetypes.init()
media_files = tuple([x for x in mimetypes.types_map if mimetypes.types_map[x].split('/')[0] in ['image', 'video', 'audio']])
filter_strings = ['/email-protection#']


def transformApi(api_key=''):
    if api_key==os.getenv("TEMP_PWD"):
        return os.getenv("OPENAI_API_KEY")
    elif api_key is None or api_key=='':
        return 'Null'
    else:
        return api_key

def get_hyperlinks(url):
    try:
        reqs = requests.get(url)
        if not reqs.headers.get('Content-Type').startswith("text/html") or 400<=reqs.status_code<600:
            return []
        soup = BeautifulSoup(reqs.text, 'html.parser')
    except Exception as e:
        print(e)
        return []
    
    hyperlinks = []
    for link in soup.find_all('a', href=True):
        hyperlinks.append(link.get('href'))

    return hyperlinks


# Function to get the hyperlinks from a URL that are within the same domain
def get_domain_hyperlinks(local_domain, url):
    clean_links = []
    for link in set(get_hyperlinks(url)):
        clean_link = None

        # If the link is a URL, check if it is within the same domain
        if re.search(HTTP_URL_PATTERN, link):
            # Parse the URL and check if the domain is the same
            url_obj = urlparse(link)
            if url_obj.netloc.replace('www.','') == local_domain.replace('www.',''):
                clean_link = link

        # If the link is not a URL, check if it is a relative link
        else:
            if link.startswith("/"):
                link = link[1:]
            elif link.startswith(("#", '?', 'mailto:')):
                continue

            if 'wp-content/uploads' in url:
                clean_link = url+ "/" + link
            else:
                clean_link = "https://" + local_domain + "/" + link

        if clean_link is not None:
            clean_link = clean_link.strip().rstrip('/').replace('/../', '/')

            if not any(x in clean_link for x in filter_strings):
                clean_links.append(clean_link)

    # Return the list of hyperlinks that are within the same domain
    return list(set(clean_links))

# this function will get you a list of all the URLs from the base URL
def crawl(url, local_domain, prog=None):
    # Create a queue to store the URLs to crawl
    queue = deque([url])

    # Create a set to store the URLs that have already been seen (no duplicates)
    seen = set([url])

    # While the queue is not empty, continue crawling
    while queue:
        # Get the next URL from the queue
        url_pop = queue.pop()
        # Get the hyperlinks from the URL and add them to the queue
        for link in get_domain_hyperlinks(local_domain, url_pop):
            if link not in seen:
                queue.append(link)
                seen.add(link)
                if len(seen)>=100:
                    return seen
        if prog is not None: prog(1, desc=f'Crawling: {url_pop}')
    
    return seen


def ingestURL(documents, url, crawling=True, prog=None):
    url = url.rstrip('/')
    # Parse the URL and get the domain
    local_domain = urlparse(url).netloc
    if not (local_domain and url.startswith('http')):
        return documents
    print('Loading URL', url)
    if crawling:
        # crawl to get other webpages from this URL
        if prog is not None: prog(0, desc=f'Crawling: {url}')
        links = crawl(url, local_domain, prog)
        if prog is not None: prog(1, desc=f'Crawling: {url}')
    else:
        links = set([url])
    # separate pdf and other links
    c_links, pdf_links = [], []
    for x in links:
        if x.endswith('.pdf'):
            pdf_links.append(x)
        elif not x.endswith(media_files):
            c_links.append(x)

    #  Clean links loader using WebBaseLoader
    if prog is not None: prog(0.5, desc=f'Ingesting: {url}')
    if c_links:
        loader = WebBaseLoader(list(c_links))
        documents.extend(loader.load())

    # remote PDFs loader
    for pdf_link in list(pdf_links):
        loader = PyMuPDFLoader(pdf_link)
        doc = loader.load()
        for x in doc:
            x.metadata['source'] = loader.source
        documents.extend(doc)

    return documents

def ingestFiles(documents, files_list, prog=None):
    for fPath in files_list:
        doc = None
        if fPath.endswith('.pdf'):
            doc = PyMuPDFLoader(fPath).load()
        elif fPath.endswith('.txt') and not 'WhatsApp Chat with' in fPath:
            doc = TextLoader(fPath).load()
        elif fPath.endswith(('.doc', 'docx')):
            doc = Docx2txtLoader(fPath).load()
        elif 'WhatsApp Chat with' in fPath and fPath.endswith('.csv'): # Convert Whatsapp TXT files to CSV using https://whatstk.streamlit.app/
            doc = WhatsAppChatLoader(fPath).load()
        else:
            pass
        
        if doc is not None and doc[0].page_content:
            if prog is not None: prog(1, desc='Loaded file: '+fPath.rsplit('/')[0])
            print('Loaded file:', fPath)
            documents.extend(doc)
    return documents


def data_ingestion(inputDir=None, file_list=[], url_list=[], prog=None):
    documents = []
    # Ingestion from Input Directory
    if inputDir is not None:
        files = [str(x) for x in Path(inputDir).glob('**/*')]
        documents = ingestFiles(documents, files)
    if file_list:
        documents = ingestFiles(documents, file_list, prog)
    # Ingestion from URLs - also try https://python.langchain.com/docs/integrations/document_loaders/recursive_url_loader
    if url_list:
        for url in url_list:
            documents = ingestURL(documents, url, prog=prog)        

    # Cleanup documents
    for x in documents:
        if 'WhatsApp Chat with' not in x.metadata['source']:
            x.page_content = x.page_content.strip().replace('\n', ' ').replace('\\n', ' ').replace('  ', ' ')
    
    # print(f"Total number of documents: {len(documents)}")
    return documents


def split_docs(documents):
    # Splitting and Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=250) # default chunk size of 4000 makes around 1k tokens per doc. with k=4, this means 4k tokens input to LLM.
    docs = text_splitter.split_documents(documents)
    return docs


def getSourcesFromMetadata(metadata, sourceOnly=True, sepFileUrl=True):
    # metadata: list of metadata dict from all documents
    setSrc = set()
    for x in metadata:
        metadataText = '' # we need to convert each metadata dict into a string format. This string will be added to a set
        if x is not None:
            # extract source first, and then extract all other items
            source = x['source']
            source = source.rsplit('/',1)[-1] if 'http' not in source else source
            notSource = []
            for k,v in x.items():
                    if v is not None and k!='source' and k in ['page', 'title']:
                        notSource.extend([f"{k}: {v}"])
            metadataText = ', '.join([f'source: {source}'] + notSource) if sourceOnly==False else source
            setSrc.add(metadataText)

    if sepFileUrl:
        src_files = '\n'.join(([f"{i+1}) {x}" for i,x in enumerate(sorted([x for x in setSrc if 'http' not in x], key=str.casefold))]))
        src_urls = '\n'.join(([f"{i+1}) {x}" for i,x in enumerate(sorted([x for x in setSrc if 'http' in x], key=str.casefold))]))

        src_files = 'Files:\n'+src_files if src_files else ''
        src_urls  = 'URLs:\n'+src_urls if src_urls else ''
        newLineSep = '\n\n' if src_files and src_urls else ''
        
        return src_files + newLineSep + src_urls , len(setSrc)
    else:
        src_docs = '\n'.join(([f"{i+1}) {x}" for i,x in enumerate(sorted(list(setSrc), key=str.casefold))]))
        return src_docs, len(setSrc)
    

def getVsDict(embeddingFunc, docs, vsDict={}):
    # create chroma client if doesnt exist
    if vsDict.get('chromaClient') is None:
        vsDict['chromaDir'] = './vecstore/'+str(uuid.uuid1())
        vsDict['chromaClient'] = Chroma(embedding_function=embeddingFunc, persist_directory=vsDict['chromaDir'])
    # clear chroma client before adding new docs
    if vsDict['chromaClient']._collection.count()>0:
        vsDict['chromaClient'].delete(vsDict['chromaClient'].get()['ids'])
    # add new docs to chroma client
    vsDict['chromaClient'].add_documents(docs)
    print('vectorstore count:',vsDict['chromaClient']._collection.count(), 'at', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    return vsDict

# used for Hardcoded documents only - not uploaded by user (userData_vecStore is separate function)
def localData_vecStore(openApiKey=None, inputDir=None, file_list=[], url_list=[], vsDict={}):
    documents = data_ingestion(inputDir, file_list, url_list)
    if not documents:
       return {}
    docs = split_docs(documents)
    # Embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openApiKey)
    # create chroma client if doesnt exist
    vsDict_hd = getVsDict(embeddings, docs, vsDict)
    # get sources from metadata
    src_str = getSourcesFromMetadata(vsDict_hd['chromaClient'].get()['metadatas'])
    src_str = str(src_str[1]) + ' source document(s) successfully loaded in vector store.'+'\n\n' + src_str[0]
    print(src_str)
    return vsDict_hd


def num_tokens_from_string(string, encoding_name = "cl100k_base"):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def getPersonalBotApiKey():
    print('funcCalled')
    if os.getenv("OPENAI_API_KEY"):
        return os.getenv("OPENAI_API_KEY")
    elif os.getenv("WX_API_KEY"):
        wxCreds = {'credentials' : {"url": "https://us-south.ml.cloud.ibm.com", "apikey": os.getenv("WX_API_KEY") },
                    'project_id': os.getenv("WX_PROJECT_ID")
                    }
        return wxCreds
    else:
        return None