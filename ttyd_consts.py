from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
import os
from dotenv import load_dotenv
load_dotenv()

exp_query = 'Generate top 5 questions that I can ask about this data. Questions should be very precise and short, ideally less than 10 words.'

waitText_initialize = 'Preparing the documents, please wait...'

# initialize_prompt = """Write a short welcome message to the user. Describe the data with a comprehensive overview including short summary.\
#  If this data is about a person, mention his name instead of using pronouns. After describing the overview, you should mention top 3 example questions that the user can ask about this data.\
#  \n\nYour response should be short and precise. Format of your response should be Summary:\n{Description and Summary} \n\n Example Questions:\n{Example Questions}"""

initialize_prompt = """
Describe an overview of this data. Also generate 3 example questions that can be asked about this data.
"""

user_avatar = 'https://cdn-icons-png.flaticon.com/512/6861/6861326.png'
# user_avatar = None
bot_avatar = 'https://cdn-icons-png.flaticon.com/512/1782/1782384.png'

nustian_exps = ['Tell me about NUSTIAN',
                'Who is the NUSTIAN regional lead for Silicon Valley?',
                'Tell me details about NUSTIAN coaching program.',
                'How can we donate to NUSTIAN fundraiser?',
                'Who is the president of NUSTIAN?',
                "What are top five missions of NUSTIAN?",
            ]

stdlQs_rb_info = 'Standalone question is a new rephrased question generated based on your original question and chat history'

stdlQs_rb_choices =  ['Retrieve relavant docs using original question, send original question to LLM (Chat history not considered)'\
                    , 'Retrieve relavant docs using standalone question, send original question to LLM'\
                    , 'Retrieve relavant docs using standalone question, send standalone question to LLM']


bam_models = sorted(['bigscience/bloom',
 'salesforce/codegen2-16b',
 'codellama/codellama-34b-instruct',
 'tiiuae/falcon-40b',
 'ibm/falcon-40b-8lang-instruct',
 'google/flan-t5-xl',
 'google/flan-t5-xxl',
 'google/flan-ul2',
 'eleutherai/gpt-neox-20b',
 'togethercomputer/gpt-neoxt-chat-base-20b',
 'ibm/granite-13b-sft',
 'ibm/granite-13b-sft-cft',
 'ibm/granite-3b-code-v1',
 'meta-llama/llama-2-13b',
 'meta-llama/llama-2-13b-chat',
 'meta-llama/llama-2-13b-chat-beam',
 'meta-llama/llama-2-70b',
 'meta-llama/llama-2-70b-chat',
 'meta-llama/llama-2-7b',
 'meta-llama/llama-2-7b-chat',
 'mosaicml/mpt-30b',
 'ibm/mpt-7b-instruct',
 'bigscience/mt0-xxl',
 'bigcode/starcoder',
 'google/ul2'])

model_dd_info = 'Make sure your credentials are submitted before changing the model. You can also input any OpenAI model name or Watsonx/BAM model ID.' 

model_dd_choices = ['gpt-3.5-turbo (openai)', 'gpt-3.5-turbo-16k (openai)', 'gpt-4 (openai)', 'text-davinci-003 (Legacy - openai)', 'text-curie-001 (Legacy - openai)', 'babbage-002 (openai)'] + [model.value+' (watsonx)' for model in ModelTypes] + [model + ' (bam)' for model in bam_models]


OaiDefaultModel = 'gpt-3.5-turbo (openai)'
WxDefaultModel = 'meta-llama/llama-2-70b-chat (watsonx)'
BamDefaultModel =  'meta-llama/llama-2-70b-chat (bam)'


url_tb_info = 'Upto 100 domain webpages will be crawled for each URL. You can also enter online PDF files.'

url_tb_ph = 'https://example.com, https://another.com, https://anyremotedocument.pdf'


md_title_general = """
    ## Chat with your documents and websites<br>
    Step 1) Enter your credentials, and click Submit.<br>
    Step 2) Upload your documents and/or enter URLs, then click Load Data.<br>
    Step 3) Once data is loaded, click Initialize Chatbot (at the bottom of the page) to start talking to your data.<br>

    Your documents should be semantically similar (covering related topics or having the similar meaning) in order to get the best results.
    You may also play around with Advanced Settings, like changing the model name and parameters.
    """

md_title_nustian = """
    ## Chat with NUSTIAN website<br>
    Step 1) Submit your credentials.<br>
    Step 2) Click Initialize Chatbot to start sending messages.<br>

    You may also play around with Advanced Settings, like changing the model name and parameters.
    """

md_title_arslan = """
    ## Talk to Arslan<br>
    Welcome to Arslan Ahmed's Chatbot!<br>
    This is LLM-based question-answer application built using Retrieval Augmented Generation (RAG) approach with Langchain, implementing Generative AI technology.\
    He has developed this application to help people get quick answers on frequently asked questions and topics, rather than waiting for his personal reply.\
    Currently, this chatbot is trained on Arslan's resume and LinkedIn profile, with plans to incorporate additional data in the future.<br><br>
    By default, this chatbot is powered by OpenAI's Large Language Model gpt-3.5-turbo. For those interested to explore, there are options under Advanced Settings to change the model and its parameters.
    """


welcomeMsgArslan = """Summary: The document provides a comprehensive overview of Arslan Ahmed\'s professional background and expertise as a data scientist.\
 It highlights his experience in various industries and his proficiency in a wide range of data analysis tools and techniques.\
 The document also mentions his involvement in research projects, publications, and academic achievements.\
\n\nExample Questions:
1. What are some of the key projects that Arslan has worked on as a data scientist?
2. What tools and technologies did Arslan Ahmed utilize in his data science work at IBM?
3. Tell me about Arslan's educational background.
"""

welcomeMsgDefault = """Hello and welcome! I'm your personal data assistant. Ask me anything about your data and I'll try my best to answer."""


def welcomeMsgUser(user):
    return f"""Hi, Welcome to personal chatbot of {user}. I am trained on the documents {user} has provided me. Ask me anything about {user} and I'll try my best to answer."""


gDrFolder=(os.getenv("GDRIVE_FOLDER_URL",'')).replace('?usp=sharing','')

class TtydMode():
    def __init__(self, name='', title='', type='', dir=None, files=[], urls=[], vis=False, welMsg='', def_k=4, gDrFolder=''):
        self.name = name
        self.title = title # markdown title for the top display
        self.type = type # userInputDocs, fixedDocs, personalBot
        self.inputDir=dir
        self.file_list=files
        self.url_list=urls
        self.gDriveFolder=gDrFolder
        self.uiAddDataVis = vis # load data from user - this will be true for type = userInputDocs
        self.welcomeMsg = welMsg #welcome msg constant - if not provided LLM will generate it
        self.k = def_k # default k docs to retrieve



mode_general = TtydMode(name='general', title=md_title_general, type='userInputDocs', vis=True)
mode_nustian = TtydMode(name='nustian', title=md_title_nustian, type='fixedDocs', urls=['https://nustianusa.org', 'https://nustian.ca'])
mode_arslan = TtydMode(name='arslan', title=md_title_arslan, type='personalBot', welMsg=welcomeMsgArslan, def_k=8, gDrFolder=gDrFolder)