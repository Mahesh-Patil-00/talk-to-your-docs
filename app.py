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
from ttyd_functions import *
from ttyd_consts import *

###############################################################################################

# You want to hardcode Documents or take it from UI?
UiAddData = False

if UiAddData: # take input data from UI
    md_title = md_title_general

else: # provide paths to the data
    url_list = ['https://www.nustianusa.org', 'https://www.nustian.ca']
    # local vector store as opposed to gradio state vector store
    vsDict_hard = localData_vecStore(os.getenv("OPENAI_API_KEY"), url_list=url_list)
    md_title = md_title_nustian


###############################################################################################

                                    # Gradio

###############################################################################################

def generateExamples(api_key_st, vsDict_st):
    qa_chain = RetrievalQA.from_llm(llm=ChatOpenAI(openai_api_key=api_key_st, temperature=0), 
                    retriever=vsDict_st['chromaClient'].as_retriever(search_type="similarity", search_kwargs={"k": 4}))

    result = qa_chain({'query': exp_query})
    answer = result['result'].strip('\n')
    grSamples = [[]]
    if answer.startswith('1. '):
        lines = answer.split("\n")  # split the answers into individual lines
        list_items = [line.split(". ")[1] for line in lines]  # extract each answer after the numbering
        grSamples = [[x] for x in list_items] # gr takes list of each item as a list

    return grSamples

# initialize chatbot function sets the QA Chain, and also sets/updates any other components to start chatting. updateQaChain function only updates QA chain and will be called whenever Adv Settings are updated.
def initializeChatbot(temp, k, modelName, stdlQs, api_key_st, vsDict_st, progress=gr.Progress()):
    progress(0.1, waitText_initialize)
    qa_chain_st = updateQaChain(temp, k, modelName, stdlQs, api_key_st, vsDict_st)
    progress(0.5, waitText_initialize)
    #generate welcome message
    result = qa_chain_st({'question': initialize_prompt, 'chat_history':[]})

    # exSamples = generateExamples(api_key_st, vsDict_st)
    # exSamples_vis = True if exSamples[0] else False

    return qa_chain_st, btn.update(interactive=True), initChatbot_btn.update('Chatbot ready. Now visit the chatbot Tab.', interactive=False)\
        , aKey_tb.update(), gr.Tabs.update(selected='cb'), chatbot.update(value=[('', result['answer'])])


def setApiKey(api_key):
    if api_key==os.getenv("TEMP_PWD") and os.getenv("OPENAI_API_KEY") is not None:
        api_key=os.getenv("OPENAI_API_KEY")
    try:
        # api_key='Null' if api_key is None or api_key=='' else api_key
        openai.Model.list(api_key=api_key) # test the API key
        api_key_st = api_key

        return aKey_tb.update('API Key accepted', interactive=False, type='text'), aKey_btn.update(interactive=False), api_key_st
    except Exception as e:
        return aKey_tb.update(str(e), type='text'), *[x.update() for x in [aKey_btn, api_key_state]]
    
# convert user uploaded data to vectorstore
def uiData_vecStore(userFiles, userUrls, api_key_st, vsDict_st={}, progress=gr.Progress()):
    opComponents = [data_ingest_btn, upload_fb, urls_tb]
    # parse user data
    file_paths = []
    documents = []
    if userFiles is not None:
        if not isinstance(userFiles, list): userFiles = [userFiles]
        file_paths = [file.name for file in userFiles]
    userUrls = [x.strip() for x in userUrls.split(",")] if userUrls else []
    #create documents
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
        # api_key_st='Null' if api_key_st is None or api_key_st=='' else api_key_st
        openai.Model.list(api_key=api_key_st) # test the API key
        embeddings = OpenAIEmbeddings(openai_api_key=api_key_st)
    except Exception as e:
        return {}, str(e), *[x.update() for x in opComponents]
    
    progress(0.5, 'Creating Vector Database')
    vsDict_st = getVsDict(embeddings, docs, vsDict_st)
    # get sources from metadata
    src_str = getSourcesFromMetadata(vsDict_st['chromaClient'].get()['metadatas'])
    src_str = str(src_str[1]) + ' source document(s) successfully loaded in vector store.'+'\n\n' + src_str[0]
    
    progress(1, 'Data loaded')
    return vsDict_st, src_str, *[x.update(interactive=False) for x in [data_ingest_btn, upload_fb]], urls_tb.update(interactive=False, placeholder='')

# just update the QA Chain, no updates to any UI
def updateQaChain(temp, k, modelName, stdlQs, api_key_st, vsDict_st):
    # if we are not adding data from ui, then use vsDict_hard as vectorstore
    if vsDict_st=={} and not UiAddData: vsDict_st=vsDict_hard
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
    gr.Markdown(md_title)
    with gr.Tabs() as tabs:
        with gr.Tab('Initialization', id='init'):
            with gr.Row():
                with gr.Column():
                    aKey_tb = gr.Textbox(label="OpenAI API Key", type='password'\
                            , info='You can find OpenAI API key at https://platform.openai.com/account/api-keys'\
                            , placeholder='Enter your API key here and hit enter to begin chatting')
                    aKey_btn = gr.Button("Submit API Key")     
            with gr.Row(visible=UiAddData):
                upload_fb = gr.Files(scale=5, label="Upload (multiple) Files - pdf/txt/docx supported", file_types=['.doc', '.docx', 'text', '.pdf', '.csv'])
                urls_tb = gr.Textbox(scale=5, label="Enter URLs starting with https (comma separated)"\
                                    , info=url_tb_info\
                                    , placeholder=url_tb_ph)
                data_ingest_btn = gr.Button("Load Data")
            status_tb = gr.TextArea(label='Status bar', show_label=False, visible=UiAddData)
            initChatbot_btn = gr.Button("Initialize Chatbot", variant="primary")

        with gr.Tab('Chatbot', id='cb'):
            with gr.Row():
                chatbot = gr.Chatbot(label="Chat History", scale=2)
                srcDocs = gr.TextArea(label="References")
            msg = gr.Textbox(label="User Input",placeholder="Type your questions here")
            with gr.Row():
                btn = gr.Button("Send Message", interactive=False, variant="primary")
                clear = gr.ClearButton(components=[msg, chatbot, srcDocs], value="Clear chat history")
            # exp_comp = gr.Dataset(scale=0.7, samples=[['123'],['456'], ['123'],['456'],['456']], components=[msg], label='Examples (auto generated by LLM)', visible=False)
            # gr.Examples(examples=exps,  inputs=msg)
            with gr.Accordion("Advance Settings - click to expand", open=False):
                with gr.Row():
                    with gr.Column():
                        temp_sld = gr.Slider(minimum=0, maximum=1, step=0.1, value=0.7, label="Temperature", info='Sampling temperature to use when calling LLM. Defaults to 0.7')
                        k_sld = gr.Slider(minimum=1, maximum=10, step=1, value=4, label="K", info='Number of relavant documents to return from Vector Store. Defaults to 4')
                        model_dd = gr.Dropdown(label='Model Name'\
                                , choices=model_dd_choices\
                                , value=model_dd_choices[0], allow_custom_value=True\
                                , info=model_dd_info)
                    stdlQs_rb = gr.Radio(label='Standalone Question', info=stdlQs_rb_info\
                            , type='index', value=stdlQs_rb_choices[1]\
                            , choices=stdlQs_rb_choices)
                   
    ### Setup the Gradio Event Listeners

    # API button
    aKey_btn_args = {'fn':setApiKey, 'inputs':[aKey_tb], 'outputs':[aKey_tb, aKey_btn, api_key_state]}
    aKey_btn.click(**aKey_btn_args)
    aKey_tb.submit(**aKey_btn_args)

    # Data Ingest Button
    data_ingest_btn.click(uiData_vecStore, [upload_fb, urls_tb, api_key_state, chromaVS_state], [chromaVS_state, status_tb, data_ingest_btn, upload_fb, urls_tb])

    # Adv Settings
    advSet_args = {'fn':updateQaChain, 'inputs':[temp_sld, k_sld, model_dd, stdlQs_rb, api_key_state, chromaVS_state], 'outputs':[qa_state]}
    temp_sld.release(**advSet_args)
    k_sld.release(**advSet_args)
    model_dd.change(**advSet_args)
    stdlQs_rb.change(**advSet_args)
    
    # Initialize button
    initChatbot_btn.click(initializeChatbot, [temp_sld, k_sld, model_dd, stdlQs_rb, api_key_state, chromaVS_state], [qa_state, btn, initChatbot_btn, aKey_tb, tabs, chatbot])

    # Chatbot submit button
    chat_btn_args = {'fn':respond, 'inputs':[msg, chatbot,  qa_state], 'outputs':[msg, chatbot, srcDocs, btn]}
    btn.click(**chat_btn_args)
    msg.submit(**chat_btn_args)

demo.queue()
demo.launch(show_error=True)