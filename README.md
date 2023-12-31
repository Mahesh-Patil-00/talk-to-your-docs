---
title: Talk To Your Docs
emoji: 🌍
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 3.42.0
app_file: app.py
pinned: false
license: apache-2.0
---

<!-- Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference -->

# Talk To Your Docs

## For users
You can use this app via Hugging Face where it is already deployed:<br/>
https://huggingface.co/spaces/arslan-ahmed/talk-to-your-docs


## For developers
Source code:<br/>
https://huggingface.co/spaces/arslan-ahmed/talk-to-your-docs/tree/main

## Create your personal bot (let people talk to your docs)
You can develop and deploy your own personal chatbot (similar to https://huggingface.co/spaces/arslan-ahmed/talk-to-arslan), with the following three commands:


docker pull arslan2k12/ttyd_base (https://hub.docker.com/r/arslan2k12/ttyd_base) <br/>
docker pull arslan2k12/arslanbot (https://hub.docker.com/r/arslan2k12/arslanbot)<br/>
docker run --rm -d -p 7860:7860 --env-file ./.env arslan2k12/ttyd_arslanbot


Contents of `.env` file:
```
TTYD_MODE=personalBot_John
#replace John with your name - use only alphabets, no special characters

GDRIVE_FOLDER_URL=https://drive.google.com/drive/folders/1ce1n1kleS1FOotdcu5joXeSRu_xnHjDt
# replace with your Google Drive folder URL that has all your knowledge base files (.pdf, .docs, .txt) - make sure this folder is publicly accessible (everyone with the link)

OPENAI_API_KEY=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
# your OpenAI API Key taken from https://platform.openai.com/account/api-keys
```