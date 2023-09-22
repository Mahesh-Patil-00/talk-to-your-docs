from dotenv import load_dotenv
import datetime
import openai
import uuid
import gradio as gr
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain.embeddings import SentenceTransformerEmbeddings

import os
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI
from langchain.document_loaders import WebBaseLoader, TextLoader, Docx2txtLoader, PyMuPDFLoader
from whatsapp_chat_custom import WhatsAppChatLoader # use this instead of from langchain.document_loaders import WhatsAppChatLoader

from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM

import genai

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

load_dotenv()
TTYD_MODE = os.getenv("TTYD_MODE",'')


# select the mode when starting container - modes options are in ttyd_consts.py
if TTYD_MODE.split('_')[0]=='personalBot':
    mode = mode_arslan
    if TTYD_MODE!='personalBot_Arslan':
        user = TTYD_MODE.split('_')[1]
        mode.title='## Talk to '+user
        mode.welcomeMsg= welcomeMsgUser(user)

elif os.getenv("TTYD_MODE",'')=='nustian':
    mode = mode_nustian
else:
    mode = mode_general


if mode.type!='userInputDocs':
    # local vector store as opposed to gradio state vector store, if we the user is not uploading the docs
    vsDict_hard = localData_vecStore(getPersonalBotApiKey(), inputDir=mode.inputDir, file_list=mode.file_list, url_list=mode.url_list, gGrUrl=mode.gDriveFolder)

###############################################################################################

                                    # Gradio

###############################################################################################

def setOaiApiKey(creds):
    creds = getOaiCreds(creds)
    try:
        openai.Model.list(api_key=creds.get('oai_key','Null')) # test the API key
        api_key_st = creds
        return 'OpenAI credentials accepted.', *[x.update(interactive=False) for x in credComps_btn_tb], api_key_st
    except Exception as e:
        gr.Warning(str(e))
        return [x.update() for x in credComps_op]

def setBamApiKey(creds):
    creds = getBamCreds(creds)
    try:
        genai.Model.models(credentials=creds['bam_creds'])
        api_key_st = creds
        return 'BAM credentials accepted.', *[x.update(interactive=False) for x in credComps_btn_tb], api_key_st
    except Exception as e:
        gr.Warning(str(e))
        return [x.update() for x in credComps_op]

def setWxApiKey(key, p_id):
    creds = getWxCreds(key, p_id)
    try:
        Model(model_id='google/flan-ul2', credentials=creds['credentials'], project_id=creds['project_id']) # test the API key
        api_key_st = creds
        return 'Watsonx credentials accepted.', *[x.update(interactive=False) for x in credComps_btn_tb], api_key_st
    except Exception as e:
        gr.Warning(str(e))
        return [x.update() for x in credComps_op]
    

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
        gr.Error('No documents found')
        return {}, '', *[x.update() for x in opComponents]
    # Splitting and Chunks
    docs = split_docs(documents)
    # Embeddings
    try:
        embeddings = getEmbeddingFunc(api_key_st)
    except Exception as e:
        gr.Error(str(e))
        return {}, '', *[x.update() for x in opComponents]
    
    progress(0.5, 'Creating Vector Database')
    vsDict_st = getVsDict(embeddings, docs, vsDict_st)
    # get sources from metadata
    src_str = getSourcesFromMetadata(vsDict_st['chromaClient'].get()['metadatas'])
    src_str = str(src_str[1]) + ' source document(s) successfully loaded in vector store.'+'\n\n' + src_str[0]
    
    progress(1, 'Data loaded')
    return vsDict_st, src_str, *[x.update(interactive=False) for x in [data_ingest_btn, upload_fb]], urls_tb.update(interactive=False, placeholder='')

# initialize chatbot function sets the QA Chain, and also sets/updates any other components to start chatting. updateQaChain function only updates QA chain and will be called whenever Adv Settings are updated.
def initializeChatbot(temp, k, modelNameDD, stdlQs, api_key_st, vsDict_st, progress=gr.Progress()):
    progress(0.1, waitText_initialize)
    chainTuple = updateQaChain(temp, k, modelNameDD, stdlQs, api_key_st, vsDict_st)
    qa_chain_st = chainTuple[0]
    progress(0.5, waitText_initialize)
    #generate welcome message
    if mode.welcomeMsg:
        welMsg = mode.welcomeMsg
    else:
        welMsg = welcomeMsgDefault
    print('Chatbot initialized at ', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    return qa_chain_st, chainTuple[1], btn.update(interactive=True), initChatbot_btn.update('Chatbot ready. Now visit the chatbot Tab.', interactive=False)\
        , status_tb.update(), gr.Tabs.update(selected='cb'), chatbot.update(value=[('', welMsg)])

# just update the QA Chain, no updates to any UI
def updateQaChain(temp, k, modelNameDD, stdlQs, api_key_st, vsDict_st):
    # if we are not adding data from ui, then use vsDict_hard as vectorstore
    if vsDict_st=={} and mode.type!='userInputDocs': vsDict_st=vsDict_hard
    
    if api_key_st['service']=='openai':
        if not 'openai' in modelNameDD:
            modelNameDD = changeModel(modelNameDD, OaiDefaultModel)
        llm = getOaiLlm(temp, modelNameDD, api_key_st)
    elif api_key_st['service']=='watsonx':
        if not 'watsonx' in modelNameDD:
            modelNameDD = changeModel(modelNameDD, WxDefaultModel)
        llm = getWxLlm(temp, modelNameDD, api_key_st)
    elif api_key_st['service']=='bam':
        if not 'bam' in modelNameDD:
            modelNameDD = changeModel(modelNameDD, BamDefaultModel)
        llm = getBamLlm(temp, modelNameDD, api_key_st)
    else:
        raise Exception('Error: Invalid or None Credentials')
    # settingsUpdated = 'Settings updated:'+ ' Model=' + modelName + ', Temp=' + str(temp)+ ', k=' + str(k)
    # gr.Info(settingsUpdated)
    
    if 'meta-llama/llama-2' in modelNameDD:
        prompt = promptLlama
    else:
        prompt = None

    # Now create QA Chain using the LLM
    if stdlQs==0: # 0th index i.e. first option
        qa_chain_st = RetrievalQA.from_llm(
                    llm=llm, 
                    retriever=vsDict_st['chromaClient'].as_retriever(search_type="similarity", search_kwargs={"k": int(k)}),
                    return_source_documents=True,
                    prompt=prompt,
                    input_key = 'question', output_key='answer' # to align with ConversationalRetrievalChain for downstream functions
                )
    else:
        rephQs = False if stdlQs==1 else True
        qa_chain_st = ConversationalRetrievalChain.from_llm(
                    llm=llm, 
                    retriever=vsDict_st['chromaClient'].as_retriever(search_type="similarity", search_kwargs={"k": int(k)}),
                    rephrase_question=rephQs,
                    return_source_documents=True,
                    return_generated_question=True,
                    combine_docs_chain_kwargs={'prompt':promptLlama}
                )
    
    return qa_chain_st, model_dd.update(value=modelNameDD)
        

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
    api_key_state = gr.State(getPersonalBotApiKey() if mode.type=='personalBot' else {}) # can be string (OpenAI) or dict (WX)
    chromaVS_state = gr.State({})


    # Setup the Gradio Layout
    gr.Markdown(mode.title)
    with gr.Tabs() as tabs:
        with gr.Tab('Initialization', id='init'):
            with gr.Row():
                with gr.Column():
                    oaiKey_tb = gr.Textbox(label="OpenAI API Key", type='password'\
                            , info='You can find OpenAI API key at https://platform.openai.com/account/api-keys')
                    oaiKey_btn = gr.Button("Submit OpenAI API Key")
                with gr.Column():
                    with gr.Row():
                        wxKey_tb = gr.Textbox(label="Watsonx API Key", type='password'\
                                , info='You can find IBM Cloud API Key at Manage > Access (IAM) > API keys on https://cloud.ibm.com/iam/overview')
                        wxPid_tb = gr.Textbox(label="Watsonx Project ID"\
                                , info='You can find Project ID at Project -> Manage -> General -> Details on https://dataplatform.cloud.ibm.com/wx/home')
                    wxKey_btn = gr.Button("Submit Watsonx Credentials")
                with gr.Column():
                    bamKey_tb = gr.Textbox(label="BAM API Key", type='password'\
                            , info='Internal IBMers only')
                    bamKey_btn = gr.Button("Submit BAM API Key")
            with gr.Row(visible=mode.uiAddDataVis):
                upload_fb = gr.Files(scale=5, label="Upload (multiple) Files - pdf/txt/docx supported", file_types=['.doc', '.docx', 'text', '.pdf', '.csv', '.ppt', '.pptx'])
                urls_tb = gr.Textbox(scale=5, label="Enter URLs starting with https (comma separated)"\
                                    , info=url_tb_info\
                                    , placeholder=url_tb_ph)
                data_ingest_btn = gr.Button("Load Data")
            status_tb = gr.TextArea(label='Status Info')
            initChatbot_btn = gr.Button("Initialize Chatbot", variant="primary")

        credComps_btn_tb = [oaiKey_tb, oaiKey_btn, bamKey_tb, bamKey_btn, wxKey_tb, wxPid_tb, wxKey_btn]
        credComps_op = [status_tb] + credComps_btn_tb + [api_key_state]
        
        with gr.Tab('Chatbot', id='cb'):
            with gr.Row():
                chatbot = gr.Chatbot(label="Chat History", scale=2, avatar_images=(user_avatar, bot_avatar))
                srcDocs = gr.TextArea(label="References")
            msg = gr.Textbox(label="User Input",placeholder="Type your questions here")
            with gr.Row():
                btn = gr.Button("Send Message", interactive=False, variant="primary")
                clear = gr.ClearButton(components=[msg, chatbot, srcDocs], value="Clear chat history")
            with gr.Accordion("Advance Settings - click to expand", open=False):
                with gr.Row():
                    with gr.Column():
                        temp_sld = gr.Slider(minimum=0, maximum=1, step=0.1, value=0.7, label="Temperature", info='Sampling temperature to use when calling LLM. Defaults to 0.7')
                        k_sld = gr.Slider(minimum=1, maximum=10, step=1, value=mode.k, label="K", info='Number of relavant documents to return from Vector Store. Defaults to 4')
                        model_dd = gr.Dropdown(label='Model Name'\
                                , choices=model_dd_choices, allow_custom_value=True\
                                , info=model_dd_info)
                    stdlQs_rb = gr.Radio(label='Standalone Question', info=stdlQs_rb_info\
                            , type='index', value=stdlQs_rb_choices[1]\
                            , choices=stdlQs_rb_choices)
                   
    ### Setup the Gradio Event Listeners

    # OpenAI API button
    oaiKey_btn_args = {'fn':setOaiApiKey, 'inputs':[oaiKey_tb], 'outputs':credComps_op}
    oaiKey_btn.click(**oaiKey_btn_args)
    oaiKey_tb.submit(**oaiKey_btn_args)

    # BAM API button
    bamKey_btn_args = {'fn':setBamApiKey, 'inputs':[bamKey_tb], 'outputs':credComps_op}
    bamKey_btn.click(**bamKey_btn_args)
    bamKey_tb.submit(**bamKey_btn_args)

    # Watsonx Creds button
    wxKey_btn_args = {'fn':setWxApiKey, 'inputs':[wxKey_tb, wxPid_tb], 'outputs':credComps_op}
    wxKey_btn.click(**wxKey_btn_args)

    # Data Ingest Button
    data_ingest_event = data_ingest_btn.click(uiData_vecStore, [upload_fb, urls_tb, api_key_state, chromaVS_state], [chromaVS_state, status_tb, data_ingest_btn, upload_fb, urls_tb])

    # Adv Settings
    advSet_args = {'fn':updateQaChain, 'inputs':[temp_sld, k_sld, model_dd, stdlQs_rb, api_key_state, chromaVS_state], 'outputs':[qa_state, model_dd]}
    temp_sld.release(**advSet_args)
    k_sld.release(**advSet_args)
    model_dd.change(**advSet_args)
    stdlQs_rb.change(**advSet_args)

    # Initialize button
    initCb_args = {'fn':initializeChatbot, 'inputs':[temp_sld, k_sld, model_dd, stdlQs_rb, api_key_state, chromaVS_state], 'outputs':[qa_state, model_dd, btn, initChatbot_btn, status_tb, tabs, chatbot]}
    if mode.type=='personalBot':
        demo.load(**initCb_args) # load Chatbot UI directly on startup
    initChatbot_btn.click(**initCb_args)

    # Chatbot submit button
    chat_btn_args = {'fn':respond, 'inputs':[msg, chatbot,  qa_state], 'outputs':[msg, chatbot, srcDocs, btn]}
    btn.click(**chat_btn_args)
    msg.submit(**chat_btn_args)

demo.queue(concurrency_count=10)
demo.launch(show_error=True)