import os


from langchain_community.vectorstores import Weaviate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_groq.chat_models import ChatGroq

import weaviate
from langchain.globals import set_llm_cache
from langchain.cache import RedisCache
# import redis

# REDIS_URL = "redis://localhost:6379/0"

# redis_client = redis.Redis.from_url(REDIS_URL)
# set_llm_cache(RedisCache(redis_client))

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
WEAVIATE_INDEX_NAME = os.getenv('WEAVIATE_INDEX_NAME')
WEAVIATE_URL = os.getenv('WEAVIATE_URL')

client = weaviate.Client(
  url=WEAVIATE_URL,
)

vectorstore = Weaviate(client, 
                       WEAVIATE_INDEX_NAME, 
                       "content")

retriever = vectorstore.as_retriever()




# RAG prompt
template = """You're an Friendly AI assistant, your name is Claro, you can make normal conversations in a friendly manner, and also provide Answer the question based on the following context make sure it sounds like human and official assistant:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)




# RAG
model = ChatGroq(api_key=GROQ_API_KEY)

chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
)



# Add typing for input
class Question(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question)