exp_query = 'Generate top 5 questions that I can ask about this data. Questions should be very precise and short, ideally less than 10 words.'

waitText_initialize = 'Preparing the documents, please wait...'

initialize_prompt = 'Write a short welcome message to the user. Describe the documents with a brief overview including short summary or any highlights.\
     If these documents are about a person, mention his name instead of using pronouns. After describing the overview, you should mention top 3 example questions that the user can ask about this data.\
    Your response should be short and precise. Format of your response should be Description:\n{description} \n\n Example Questions:\n{examples}'

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



model_dd_info = 'You can also input any OpenAI model name, compatible with /v1/completions or /v1/chat/completions endpoint. Details: https://platform.openai.com/docs/models/'

model_dd_choices = ['gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-4', 'text-davinci-003 (Legacy)', 'text-curie-001 (Legacy)', 'babbage-002']

url_tb_info = 'Upto 100 domain webpages will be crawled for each URL. You can also enter online PDF files.'

url_tb_ph = 'https://example.com, https://another.com, https://anyremotedocument.pdf'


md_title_general = """
    ## Chat with your documents and websites<br>
    Step 1) Enter your OpenAI API Key, and click Submit.<br>
    Step 2) Upload your documents and/or enter URLs, then click Load Data.<br>
    Step 3) Once data is loaded, click Initialize Chatbot (at the bottom of the page) to start talking to your data.<br>

    Your documents should be semantically similar (covering related topics or having the similar meaning) in order to get the best results.
    You may also play around with Advanced Settings, like changing the model name and parameters.
    """

md_title_nustian = """
    ## Chat with NUSTIAN website<br>
    Step 1) Submit your OpenAI API Key.<br>
    Step 2) Click Initialize Chatbot to start sending messages.<br>

    You may also play around with Advanced Settings, like changing the model name and parameters.
    """