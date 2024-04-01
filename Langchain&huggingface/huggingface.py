import os
os.environ['OPENAI_API_KEY'] = ''
os.environ['HUGGINGFACEHUB_API_TOKEN'] = ''

# '''
# from langchain import LLMChain
# from langchain.prompts import PromptTemplate
# from langchain.llms import HuggingFaceHub

# # HuggingFace Repository ID
# repo_id = 'mistralai/Mistral-7B-v0.1'

# # 질의내용
# question = "Who is Son Heung Min?"

# # 템플릿
# template = """Question: {question}

# Answer: """

# # 프롬프트 템플릿 생성
# prompt = PromptTemplate(template=template, input_variables=["question"])

# # HuggingFaceHub 객체 생성
# llm = HuggingFaceHub(
#     repo_id=repo_id, 
#     model_kwargs={"temperature": 0.2, 
#                   "max_length": 128}
# )

# # LLM Chain 객체 생성
# llm_chain = LLMChain(prompt=prompt, llm=llm)

# # 실행
# print(llm_chain.run(question=question))


# # HuggingFace Repository ID
# repo_id = 'google/flan-t5-xxl'

# # 질의내용
# question = "who is Son Heung Min?"

# # 템플릿
# template = """Question: {question}

# Answer: """

# # 프롬프트 템플릿 생성
# prompt = PromptTemplate(template=template, input_variables=["question"])

# # HuggingFaceHub 객체 생성
# llm = HuggingFaceHub(
#     repo_id=repo_id, 
#     model_kwargs={"temperature": 0.2, 
#                   "max_length": 512}
# )

# # LLM Chain 객체 생성
# llm_chain = LLMChain(prompt=prompt, llm=llm)

# # 실행
# print(llm_chain.run(question=question))
# '''

# 허깅페이스 모델/토크나이저를 다운로드 받을 경로
# (예시)
# os.environ['HF_HOME'] = '/home/jovyan/work/tmp'
'''
os.environ['HF_HOME'] = '/home/centos/daegeon/chatbot/model'
import langchain
print(langchain.__version__)

from langchain import LLMChain
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate

# HuggingFace Model ID
#model_id = 'beomi/llama-2-ko-7b'
# ranking 1~6 
model_id = 'krevas/SOLAR-10.7B'
# chihoonlee10/T3Q-ko-solar-sft-dpo-v1.0   
# chihoonlee10/T3Q-ko-solar-dpo-v3.0
# chlee10/T3Q-ko-solar-sft-v3.0 
# ENERGY-DRINK-LOVE/nox_DPOv3  # AI-hub dataset 을 활용
# hwkwon/S-SOLAR-10.7B-v1.4

# HuggingFacePipeline 객체 생성
llm = HuggingFacePipeline.from_model_id(
    model_id=model_id, 
    device=-1,               # -1: CPU(default), 0번 부터는 CUDA 디바이스 번호 지정시 GPU 사용하여 추론
    task="text-generation", # 텍스트 생성
    model_kwargs={"temperature": 0.1, 
                  "max_length": 512},
)

# 템플릿
template = """질문: {question}

답변: """

# 프롬프트 템플릿 생성
prompt = PromptTemplate.from_template(template)

# LLM Chain 객체 생성
llm_chain = LLMChain(prompt=prompt, llm=llm)

# 실행
question = "대한민국의 수도는 어디야?"
print(llm_chain.run(question=question))

# 실행
question = "캐나다의 수도와 대한민국의 수도까지의 거리는 어떻게 돼?"
print(llm_chain.run(question=question))

# 실행
question = "휴맥스모빌리티 ev충전기 사용시 충전이 되다가 멈추는 상황이 발생하는데 이유가 뭔지 어떻게 해결할 수 있는지 알려주세요."
print(llm_chain.run(question=question))
# '''
# '''
# # ref
# # python : https://python.langchain.com/en/latest/index.html
# # concept : https://docs.langchain.com/docs/
# # src : https://github.com/hwchase17/langchain
# # youtube : https://youtube.com/KdbPZNdFJU0

# # INSTALL LIBRARY
# !pip install openai
# !pip install langchain
# !pip install google-search-results
# !pip install wikipedia

# # API KEY SETTING
import os 
OPENAI_API_KEY = ''
HUGGINGFACEHUB_API_TOKEN = '' # Model download 무료
#SERPAPI_API_KEY = '' # 월 100회 무료 
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACEHUB_API_TOKEN
#os.environ['SERPAPI_API_KEY'] = ''

# 1. OPENAI LLM(text-davinci-003)
from langchain.llms import OpenAI
llm = OpenAI(model_name='gpt-3.5-turbo-0301')
llm('1980년대 메탈 음악 2곡 추천해줘')

# 2. ChatOpenAI LLM (gpt-3.5-turbo)
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

chat = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.9)
sys = SystemMessage(content="당신은 음악 추천을 해주는 전문 AI 입니다.")
msg = HumanMessage(content="1990년대 발라드 음악 3곡 추천해줘.")

aimsg = chat([sys,msg])
print(aimsg.content)

# 3. Prompt Template & Chain  
from langchain.prompts import PromptTemplate
prompt = PromptTemplate(
    input_variables=["상품"],
    template="{상품} 에 best 1,2,3 간단하게 소개해줘. 한글로 답변해줘",
)
prompt.format(상품="AI 여행 추천 서비스")

from langchain.chains import LLMChain
chain = LLMChain(llm=chat, prompt=prompt)
print(chain.run(상품="국내 여행 추천"))

# 4. ChatPromptTemplate & chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
chat = ChatOpenAI(temperature=0)
template= 'you are a helpful assisstant that translates {input_language} to {output_language}.'
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template="{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
chatchain = LLMChain(llm=chat, prompt=chat_prompt)
print(chatchain.run(input_language="English", output_language="Korean", text="I love you"))


# Agents and tools 쪽 아직 model action 관련해서 반복실행되는 issue 때문에 주석처리 (비용반복)
# 5. Agents and Tools 
# from langchain.agents import load_tools
# from langchain.agents import initialize_agent
# from langchain.agents import AgentType
# tools = load_tools(["wikipedia", "llm-math"], llm=chat)
# agent = initialize_agent(tools,llm=chat, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
# print(agent.run('페이스북 창업자가 누구야?'))
# print(agent.tools)
# print(conversation.memory)

# # 6. Memory 
# from langchain import ConversationChain
# conversation = ConversationChain(llm=chat, verbose=True)
# print(conversation.predict(input='인공지능에서 Transformer가 뭐야?'))
# print(conversation.predict(input='RNN하고 차이 설명해줘.'))
# print(conversation.memory)

# 7. Document Loaders
from langchain.document_loaders import WebBaseLoader
loader = WebBaseLoader(web_path="https://ko.wikipedia.org/wiki/NewJeans")
documents = loader.load()

from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=990, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
print(len(docs))

print(docs[1].page_content)

# summary 관련한 작업도 consise error issue 로 주석처리
# 8. Summarization 
# from langchain.chains.summarize import load_summarize_chain
# chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
# chain.run(docs[1:3])
from langchain.chat_models import ChatOpenAI
chat = ChatOpenAI(model_name='gpt-4', temperature=0)
#chat = ChatOpenAI(model_name='gpt-4', temperature=0.9)

from langchain.chains.summarize import load_summarize_chain
chain = load_summarize_chain(chat, chain_type="map_reduce", verbose=True)
chain.run(docs[1:5])

# 9. Embeddings and VectorStore - model setting  instructor embedding 을 사용한 병렬처리 & 현재 instructor-large model token 값이 맞지 않아 pass : 추후 보충예정
#from langchain.embeddings import HuggingFaceEmbeddings
#from langchain.embeddings import OpenAIEmbeddings
#from InstructorEmbedding import INSTRUCTOR
#from langchain.embeddings import HuggingFaceInstrctEmbeddings
#from langchain_community.embeddings import HuggingFaceInstructEmbeddings

#embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large",model_kwargs = {'device': 'cpu'}) 
#embeddings = HuggingFaceInstructEmbeddings(model_name="intfloat/multilingual-e5-large")
#db = Chroma.from_documents(documents=texts, embedding=embeddings, collection_name="snakes", persist_directory="db")
#embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
#vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

#from InstructorEmbedding import INSTRUCTOR
#model = INSTRUCTOR('hkunlp/instructor-large')
# sentence = "3D ActionSLAM: wearable person tracking in multi-floor environments"
# 위 예시 문장 아닌 docs 에 저장된 text 사용
#instruction = "NewJeans:"
#embeddings = model.encode([[instruction,docs]])
#print(embeddings)

# 위 info 에 대한 특징 벡터 추출 error 로 인한 기본서비스로 대체 
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings

# embeddings = OpenAIEmbeddings()
embeddings = HuggingFaceEmbeddings()

# vector db 저장
from langchain.indexes import VectorstoreIndexCreator 
from langchain.vectorstores import FAISS

index = VectorstoreIndexCreator(
    vectorstore_cls = FAISS,
    embedding=embeddings
    # text_splitter=text_splitter
    ).from_loaders([loader])

# file 로 저장
index.vectorstore.save_local('faiss-nj-ko') 

# query
print(index.query('Who is NewJeans debut members?', llm=chat, verbose=True))
print(index.query('tell me all the names of the NewJeans members', llm=chat, verbose=True))
print(index.query('Who is NewJeans debut single?', llm=chat, verbose=True))

# FAISS 벡터 db 에서 disk 불러오기
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

fdb = FAISS.load_local('faiss-nj-ko', embeddings, allow_dangerous_deserialization=True)
index2 = VectorStoreIndexWrapper(vectorstore=fdb)
print(index2.query('뉴진스의 데뷔 멤버는?', llm=chat, verbose=True))


