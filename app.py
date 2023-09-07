import datetime
import openai
import uuid
import gradio as gr
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA

import os
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI
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
            if url_obj.netloc == local_domain:
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
        elif fPath.endswith('.txt'):
            doc = TextLoader(fPath).load()
        elif fPath.endswith(('.doc', 'docx')):
            doc = Docx2txtLoader(fPath).load()
        elif 'WhatsApp Chat with' in fPath and fPath.endswith('.csv'):
            doc = WhatsAppChatLoader(fPath).load()
        else:
            pass
        
        if doc is not None and doc[0].page_content:
            if prog is not None: prog(1, desc='Loaded file: '+fPath.rsplit('/')[0])
            print('Loaded file:', fPath)
            documents.extend(doc)
    return documents


def data_ingestion(inputDir=None, file_list=[], waDir=None, url_list=[], prog=None):
    documents = []
    # Ingestion from Input Directory
    if inputDir is not None:
        files = [str(x) for x in Path(inputDir).glob('**/*')]
        documents = ingestFiles(documents, files)
    if file_list:
        documents = ingestFiles(documents, file_list, prog)
    # Ingestion of whatsapp chats - Convert Whatsapp TXT files to CSV using https://whatstk.streamlit.app/
    if waDir is not None:
        for fPath in [str(x) for x in Path(waDir).glob('**/*.csv')]:
            waDoc = WhatsAppChatLoader(fPath).load()
            if waDoc[0].page_content:
                print('Loaded whatsapp file:', fPath)
                documents.extend(waDoc)
    # Ingestion from URLs - also try https://python.langchain.com/docs/integrations/document_loaders/recursive_url_loader
    if url_list:
        for url in url_list:
            documents = ingestURL(documents, url, prog=prog)
        

    # Cleanup documents
    for x in documents:
        if 'WhatsApp Chat with ' not in x.metadata['source']:
            x.page_content = x.page_content.strip().replace('\n', ' ').replace('\\n', ' ').replace('  ', ' ')
    
    print(f"Total number of documents: {len(documents)}")
    return documents


def split_docs(documents):
    # Splitting and Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=250) # default chunk size of 4000 makes around 1k tokens per doc. with k=4, this means 4k tokens input to LLM.
    docs = text_splitter.split_documents(documents)
    return docs

# used for Hardcoded documents only - not uploaded by user
def getVectorStore(openApiKey, documents, chromaClient=None):
    docs = split_docs(documents)
    # Embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openApiKey)
    # create chroma client if doesnt exist
    if chromaClient is None:
        chromaClient = Chroma(embedding_function=embeddings)
    # clear chroma client before adding new docs
    if chromaClient._collection.count()>0:
        chromaClient.delete(chromaClient.get()['ids'])
    # add new docs to chroma client
    chromaClient.add_documents(docs)
    print('vectorstore count:',chromaClient._collection.count(), 'at', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    return chromaClient


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

def num_tokens_from_string(string, encoding_name = "cl100k_base"):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
###############################################################################################

# Hardcoded Documents

# documents = []

# # Data Ingestion - take list of documents
# documents = data_ingestion(inputDir= '../reports/',waDir = '../whatsapp-exports/')
# full_text = ''.join([x.page_content for x in documents])
# print('Full Text Len:', len(full_text), 'Num tokens:', num_tokens_from_string(full_text))

# # Embeddings
# vectorstore = getVectorStore(os.getenv("OPENAI_API_KEY"), documents)



###############################################################################################

                                    # Gradio

###############################################################################################

def generateExamples(api_key_st, vsDict_st):
    qa_chain = RetrievalQA.from_llm(llm=ChatOpenAI(openai_api_key=api_key_st, temperature=0), 
                    retriever=vsDict_st['chromaClient'].as_retriever(search_type="similarity", search_kwargs={"k": 4}))

    result = qa_chain({'query': 'Generate top 5 questions that I can ask about this data. Questions should be very precise and short, ideally less than 10 words.'})
    answer = result['result'].strip('\n')
    grSamples = [[]]
    if answer.startswith('1. '):
        lines = answer.split("\n")  # split the answers into individual lines
        list_items = [line.split(". ")[1] for line in lines]  # extract each answer after the numbering
        grSamples = [[x] for x in list_items] # gr takes list of each item as a list

    return grSamples

# initialize chatbot function sets the QA Chain, and also sets/updates any other components to start chatting. updateQaChain function only updates QA chain and will be called whenever Adv Settings are updated.
def initializeChatbot(temp, k, modelName, stdlQs, api_key_st, vsDict_st, progress=gr.Progress()):
    progress(0.1, 'Analyzing your documents, please wait...')
    qa_chain_st = updateQaChain(temp, k, modelName, stdlQs, api_key_st, vsDict_st)
    progress(0.5, 'Analyzing your documents, please wait...')
    #generate welcome message
    result = qa_chain_st({'question': 'Write a short welcome message to the user. Describe the document with a brief overview and short summary or any highlights. If this document is about a person, mention his name instead of using pronouns. After this, you should include top 3 example questions that user can ask about this data. Make sure you have got answers to those questions within the data. Your response should be short and precise. Format of your response should be Summary:  {summary} \n\n\n Example Questions:  {examples}', 'chat_history':[]})
    # exSamples = generateExamples(api_key_st, vsDict_st)
    # exSamples_vis = True if exSamples[0] else False
    
    return qa_chain_st, btn.update(interactive=True), initChatbot_btn.update('Chatbot ready. Now visit the chatbot Tab.', interactive=False)\
        , status_tb.update(), gr.Tabs.update(selected='cb'), chatbot.update(value=[('', result['answer'])])



def setApiKey(api_key):
    if api_key==os.getenv("TEMP_PWD") and os.getenv("OPENAI_API_KEY") is not None:
        api_key=os.getenv("OPENAI_API_KEY")
    try:
        api_key='Null' if api_key is None or api_key=='' else api_key
        openai.Model.list(api_key=api_key) # test the API key
        api_key_st = api_key

        return aKey_tb.update('API Key accepted', interactive=False, type='text'), aKey_btn.update(interactive=False), api_key_st
    except Exception as e:
        return aKey_tb.update(str(e), type='text'), *[x.update() for x in [aKey_btn, api_key_state]]
    
# convert user uploaded data to vectorstore
def userData_vecStore(userFiles, userUrls, api_key_st, vsDict_st={}, progress=gr.Progress()):
    opComponents = [data_ingest_btn, upload_fb, urls_tb]
    file_paths = []
    documents = []
    if userFiles is not None:
        if not isinstance(userFiles, list): userFiles = [userFiles]
        file_paths = [file.name for file in userFiles]
    userUrls = [x.strip() for x in userUrls.split(",")] if userUrls else []
    documents = data_ingestion(file_list=file_paths, url_list=userUrls, prog=progress)
    if documents:
        for file in file_paths:
            os.remove(file)
    else:
        return {}, '', *[x.update() for x in opComponents]
    
    # Splitting and Chunks
    docs = split_docs(documents)
    # Embeddings
    try:
        api_key_st='Null' if api_key_st is None or api_key_st=='' else api_key_st
        openai.Model.list(api_key=api_key_st) # test the API key
        embeddings = OpenAIEmbeddings(openai_api_key=api_key_st)
    except Exception as e:
        return {}, str(e), *[x.update() for x in opComponents]
    
    progress(0.5, 'Creating Vector Database')
    # create chroma client if doesnt exist
    if vsDict_st.get('chromaDir') is None:
        vsDict_st['chromaDir'] = str(uuid.uuid1())
        vsDict_st['chromaClient'] = Chroma(embedding_function=embeddings, persist_directory=vsDict_st['chromaDir'])
    # clear chroma client before adding new docs
    if vsDict_st['chromaClient']._collection.count()>0:
        vsDict_st['chromaClient'].delete(vsDict_st['chromaClient'].get()['ids'])
    # add new docs to chroma client
    vsDict_st['chromaClient'].add_documents(docs)
    print('vectorstore count:',vsDict_st['chromaClient']._collection.count(), 'at', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    op_docs_str = getSourcesFromMetadata(vsDict_st['chromaClient'].get()['metadatas'])
    op_docs_str = str(op_docs_str[1]) + ' document(s) successfully loaded in vector store.'+'\n\n' + op_docs_str[0]
    progress(1, 'Data loaded')
    return vsDict_st, op_docs_str, *[x.update(interactive=False) for x in [data_ingest_btn, upload_fb]], urls_tb.update(interactive=False, placeholder='')

# just update the QA Chain, no updates to any UI
def updateQaChain(temp, k, modelName, stdlQs, api_key_st, vsDict_st):
    modelName = modelName.split('(')[0].strip() # so we can provide any info in brackets
    # check if the input model is chat model or legacy model
    try:
        ChatOpenAI(openai_api_key=api_key_st, temperature=0,model_name=modelName,max_tokens=1).predict('')
        llm = ChatOpenAI(openai_api_key=api_key_st, temperature=float(temp),model_name=modelName)
    except:
        OpenAI(openai_api_key=api_key_st, temperature=0,model_name=modelName,max_tokens=1).predict('')
        llm = OpenAI(openai_api_key=api_key_st, temperature=float(temp),model_name=modelName)
    # settingsUpdated = 'Settings updated:'+ ' Model=' + modelName + ', Temp=' + str(temp)+ ', k=' + str(k)
    # gr.Info(settingsUpdated)
    
    # Now create QA Chain using the LLM
    if stdlQs==0: # 0th index i.e. first option
        qa_chain_st = RetrievalQA.from_llm(
                    llm=llm, 
                    retriever=vsDict_st['chromaClient'].as_retriever(search_type="similarity", search_kwargs={"k": int(k)}),
                    return_source_documents=True,
                    input_key = 'question', output_key='answer' # to align with ConversationalRetrievalChain for downstream functions
                )
    else:
        rephQs = False if stdlQs==1 else True
        qa_chain_st = ConversationalRetrievalChain.from_llm(
                    llm=llm, 
                    retriever=vsDict_st['chromaClient'].as_retriever(search_type="similarity", search_kwargs={"k": int(k)}),
                    rephrase_question=rephQs,
                    return_source_documents=True,
                    return_generated_question=True
                )
    
    return qa_chain_st
        

def respond(message, chat_history, qa_chain):
    result = qa_chain({'question': message, "chat_history": [tuple(x) for x in chat_history]})
    src_docs = getSourcesFromMetadata([x.metadata for x in result["source_documents"]], sourceOnly=False)[0]
    # streaming
    streaming_answer = ""
    for ele in "".join(result['answer']):
        streaming_answer += ele
        yield "", chat_history + [(message, streaming_answer)], src_docs, btn.update('Please wait...', interactive=False)
    
    chat_history.extend([(message, result['answer'])])
    yield "", chat_history, src_docs, btn.update('Send Message', interactive=True)

#####################################################################################################

with gr.Blocks(theme=gr.themes.Default(primary_hue='orange', secondary_hue='gray', neutral_hue='blue'), css="footer {visibility: hidden}") as demo:

    # Initialize state variables - stored in this browser session - these can only be used within input or output of .click/.submit etc, not as a python var coz they are not stored in backend, only as a frontend gradio component
    # but if you initialize it with a default value, that value will be stored in backend and accessible across all users. You can also change it with statear.value='newValue'
    qa_state = gr.State()
    api_key_state = gr.State()
    chromaVS_state = gr.State({})


    # Setup the Gradio Layout
    gr.Markdown(
    """
    ## Chat with your documents and websites<br>
    Step 1) Enter your OpenAI API Key, and click Submit.<br>
    Step 2) Upload your documents and/or enter URLs, then click Load Data.<br>
    Step 3) Once data is loaded, click Initialize Chatbot (at the bottom of the page) to start talking to your data.<br>

    Your documents should be semantically similar (covering related topics or having the similar meaning) in order to get the best results.
    You may also play around with Advanced Settings, like changing the model name and parameters.
    """)
    with gr.Tabs() as tabs:
        with gr.Tab('Initialization', id='init'):
            with gr.Row():
                with gr.Column():
                    aKey_tb = gr.Textbox(label="OpenAI API Key", type='password'\
                            , info='You can find OpenAI API key at https://platform.openai.com/account/api-keys'\
                            , placeholder='Enter your API key here and hit enter to begin chatting')
                    aKey_btn = gr.Button("Submit API Key")     
            with gr.Row():
                upload_fb = gr.Files(scale=5, label="Upload (multiple) Files - pdf/txt/docx supported", file_types=['.doc', '.docx', 'text', '.pdf', '.csv'])
                urls_tb = gr.Textbox(scale=5, label="Enter URLs starting with https (comma separated)"\
                                    , info='Upto 100 domain webpages will be crawled for each URL. You can also enter online PDF files.'\
                                    , placeholder='https://example.com, https://another.com, https://anyremotedocument.pdf')
                data_ingest_btn = gr.Button("Load Data")
            status_tb = gr.TextArea(label='Status bar', show_label=False)
            initChatbot_btn = gr.Button("Initialize Chatbot")

        with gr.Tab('Chatbot', id='cb'):
            with gr.Row():
                chatbot = gr.Chatbot(label="Chat History", scale=2)
                srcDocs = gr.TextArea(label="References")
            msg = gr.Textbox(label="User Input",placeholder="Type your questions here")
            with gr.Row():
                btn = gr.Button("Send Message", interactive=False)
                clear = gr.ClearButton(components=[msg, chatbot, srcDocs], value="Clear chat history")
            with gr.Row():
                # exp_comp = gr.Dataset(scale=0.7, samples=[['123'],['456'], ['123'],['456'],['456']], components=[msg], label='Examples (auto generated by LLM)', visible=False)
                # gr.Examples(examples=exps,  inputs=msg)
                with gr.Accordion("Advance Settings - click to expand", open=False):
                    with gr.Row():
                        temp_sld = gr.Slider(minimum=0, maximum=1, step=0.1, value=0.7, label="Temperature", info='Sampling temperature to use when calling LLM. Defaults to 0.7')
                        k_sld = gr.Slider(minimum=1, maximum=10, step=1, value=4, label="K", info='Number of relavant documents to return from Vector Store. Defaults to 4')
                        model_dd = gr.Dropdown(label='Model Name'\
                                , choices=['gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-4', 'text-davinci-003 (Legacy)', 'text-curie-001 (Legacy)', 'babbage-002']\
                                , value='gpt-3.5-turbo', allow_custom_value=True\
                                , info='You can also input any OpenAI model name, compatible with /v1/completions or /v1/chat/completions endpoint. Details: https://platform.openai.com/docs/models/')
                    stdlQs_rb = gr.Radio(label='Standalone Question', info='Standalone question is a new rephrased question generated based on your original question and chat history'\
                                , type='index', value='Retrieve relavant docs using standalone question, send original question to LLM'\
                            , choices=['Retrieve relavant docs using original question, send original question to LLM (Chat history not considered)'\
                                    , 'Retrieve relavant docs using standalone question, send original question to LLM'\
                                    , 'Retrieve relavant docs using standalone question, send standalone question to LLM'])
                   
    ### Setup the Gradio Event Listeners

    # API button
    aKey_btn_args = {'fn':setApiKey, 'inputs':[aKey_tb], 'outputs':[aKey_tb, aKey_btn, api_key_state]}
    aKey_btn.click(**aKey_btn_args)
    aKey_tb.submit(**aKey_btn_args)

    # Data Ingest Button
    data_ingest_btn.click(userData_vecStore, [upload_fb, urls_tb, api_key_state, chromaVS_state], [chromaVS_state, status_tb, data_ingest_btn, upload_fb, urls_tb])

    # Adv Settings
    advSet_args = {'fn':updateQaChain, 'inputs':[temp_sld, k_sld, model_dd, stdlQs_rb, api_key_state, chromaVS_state], 'outputs':[qa_state]}
    temp_sld.change(**advSet_args)
    k_sld.change(**advSet_args)
    model_dd.change(**advSet_args)
    stdlQs_rb.change(**advSet_args)

    # Initialize button
    initChatbot_btn.click(initializeChatbot, [temp_sld, k_sld, model_dd, stdlQs_rb, api_key_state, chromaVS_state], [qa_state, btn, initChatbot_btn, status_tb, tabs, chatbot])

    # Chatbot submit button
    chat_btn_args = {'fn':respond, 'inputs':[msg, chatbot,  qa_state], 'outputs':[msg, chatbot, srcDocs, btn]}
    btn.click(**chat_btn_args)
    msg.submit(**chat_btn_args)

demo.queue()
demo.launch(show_error=True)