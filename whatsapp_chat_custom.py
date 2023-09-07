# created custom class for WhatsAppChatLoader - because original langchain one isnt working

import re
from pathlib import Path
from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


def concatenate_rows(date: str, sender: str, text: str) -> str:
    """Combine message information in a readable format ready to be used."""
    return f"{sender} on {date}: {text}\n\n"

# def concatenate_rows(date: str, sender: str, text: str) -> str:
#     """Combine message information in a readable format ready to be used."""
#     return f"{text}\n"

class WhatsAppChatLoader(BaseLoader):
    """Load `WhatsApp` messages text file."""

    def __init__(self, path: str):
        """Initialize with path."""
        self.file_path = path

    def load(self) -> List[Document]:
        """Load documents."""
        p = Path(self.file_path)
        text_content = ""

        ignore_lines = ["This message was deleted", "<Media omitted>"]
        #########################################################################################
        # original code from langchain replaced with this code
        #########################################################################################
        # use https://whatstk.streamlit.app/ to get CSV
        import pandas as pd
        df = pd.read_csv(p)[['date', 'username', 'message']]

        for i,row in df.iterrows():
            date = row['date']
            sender = row['username']
            text = row['message']
            
            if not any(x in text for x in ignore_lines):
                text_content += concatenate_rows(date, sender, text)

        metadata = {"source": str(p)}

        return [Document(page_content=text_content.strip(), metadata=metadata)]