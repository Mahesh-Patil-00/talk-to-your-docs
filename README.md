---
title: Talk To Your Docs
emoji: 📚
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 3.42.0
app_file: app.py
pinned: false
license: apache-2.0
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference


For users, you can use this app via Hugging Face where it is already deployed:
https://huggingface.co/spaces/arslan-ahmed/talk-to-your-docs


For developers, if you want to further develop this app, or deploy using docker:
Source code:
https://huggingface.co/spaces/arslan-ahmed/talk-to-your-docs/tree/main

Docker:
docker pull arslan2k12/ttyd_base
docker pull arslan2k12/arslanbot
docker run --rm -d -p 7860:7860 -e TTYD_MODE=xxxx -e OPENAI_API_KEY=xxxx arslan2k12/arslanbot
