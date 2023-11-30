import tiktoken
import logging
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from langchain.llms import OpenAI
from langchain import PromptTemplate

from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.summarize import load_summarize_chain

from langchain.chains import AnalyzeDocumentChain
import time


def eatUrl(url_link):
    xmem_data = {}
    try:
        loader = WebBaseLoader(url_link)
        #("https://en.wikipedia.org/wiki/Internet_Protocol_version_4")
        #docs = loader.load()
        #print(docs)
        scraper = loader.scrape()
        url_title = (scraper.find('title').text)
        content = scraper.text
        #docs[0].page_content
        textSplitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=8000,chunk_overlap=0)
        text = textSplitter.split_text(content)
        print(text)
        print(len(text))
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
        chain = load_summarize_chain(llm, chain_type="stuff")

        summarize_document_chain = AnalyzeDocumentChain(combine_docs_chain=chain, text_splitter=textSplitter)

        sum_text = summarize_document_chain.run(text[0])
        link_data = f" The link {url_link} is about the following:  {sum_text}"
        print("\n\n\t")
        print(link_data)
        #save_mem(link_data,'')
        print("\tmemory printed")
        print("\n\n\t\tURL TITLE IS: ",url_title)
        xmem_data["title"] = url_title
        xmem_data["content"] = link_data
        xmem_data["tags"] = ''
        return xmem_data
    except Exception as ex:
        print(ex)
        return xmem_data
