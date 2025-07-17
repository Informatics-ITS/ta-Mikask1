import os
import re
import json
import zipfile
import io
import pandas as pd
import transformers
import torch
from typing import Dict, List, TypedDict, Any
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from tavily import TavilyClient

load_dotenv()

MODEL_ID = "GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct"
CHUNK_FOLDER = r"chunks"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
START_INDEX = 0

pipeline = transformers.pipeline(
    "text-generation",
    model=MODEL_ID,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="cuda"
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

embedding_model = SentenceTransformer('intfloat/multilingual-e5-large-instruct', device="cuda")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("uu-index")
tavily = TavilyClient(api_key=TAVILY_API_KEY)

class AgentState(TypedDict):
    question: str
    docs: List[str]
    answer: str
    feedback: Dict[str, Any]
    web_search_results: List[Dict]
    final_answer: str
    filenames: List[str]
    used_web_search: bool

def llm(messages):
    outputs = pipeline(
        messages,
        max_new_tokens=512,
        eos_token_id=terminators,
    )
    
    return outputs[0]["generated_text"][-1]['content']

def cleanup_answer(text):
    text = text.strip()
    last_period_index = text.rfind(".")
    if last_period_index != -1:
        text = text[:last_period_index + 1]

    text = re.sub(r'[\u4e00-\u9fff]+', '', text)

    return text.replace("\n", " ")

def retrieve_documents(state: AgentState) -> AgentState:
    query = state["question"]
    query_vector = embedding_model.encode(query).tolist()

    query_results = index.query(
        vector=query_vector,
        top_k=8,
        namespace="",
        include_metadata=True
    )

    docs = []
    filenames = []
    for match in query_results.matches:
        filename = match.id + ".md"
        try:
            with open(os.path.join(CHUNK_FOLDER, filename), "r", encoding="utf-8") as f:
                docs.append(f.read())
                filenames.append(filename)
        except FileNotFoundError:
            continue

    return {"docs": docs, "filenames": filenames}

def generate_answer(state: AgentState) -> AgentState:
    question = state["question"]
    docs = state["docs"]

    context = "\n\n".join(docs)[:7000]

    messages = [
        {"role": "system", "content": "Anda adalah asisten yang membantu menjawab pertanyaan menggunakan konteks yang disediakan. Jika konteks tidak berisi informasi yang diperlukan, katakan dengan jelas. Jawab dalam Bahasa Indonesia. Jawab dalam 1 paragraf."},
        {"role": "user", "content": f"Konteks: {context}\n\nPertanyaan: {question}\n\nJawaban:"}
    ]

    answer = cleanup_answer(llm(messages))
    
    return {"answer": answer}

def judge_answer(state: AgentState) -> AgentState:
    question = state["question"]
    answer = state["answer"]

    messages = [
        {"role": "system", "content": "Anda adalah seorang hakim yang mengevaluasi apakah jawaban sepenuhnya menjawab pertanyaan. PENTING: Anda harus mengembalikan \"IS_SUFFICIENT=TRUE\" jika jawaban ini cukup."},
        {"role": "user", "content": f"Pertanyaan: {question}\n\nJawaban: {answer}\n\nEvaluasi apakah jawaban ini cukup:"}
    ]

    feedback_text = llm(messages)

    if "IS_SUFFICIENT=TRUE" in feedback_text:
        return {"feedback": {
            "is_sufficient": True
        }, "final_answer": state['answer']}

    return {"feedback": {
            "is_sufficient": False
            }}

def search_web(state: AgentState) -> AgentState:
    try:
        search_results = tavily.search(
            query=state["question"],
            search_depth="advanced",
            include_domains=["id"],
            include_answer=True,
            max_results=5
        )

        if "answer" in search_results and search_results["answer"]:
            formatted_results = f"{search_results['answer']}\n\n"

            return {"web_search_results": formatted_results, "used_web_search": True}

        for i, result in enumerate(search_results.get("results", []), 1):
            formatted_results += f"Link {i}: {result.get('title', 'Untitled')}\n"
            formatted_results += f"Content: {result.get('content', 'No content')[:500]}...\n\n"

        return {"web_search_results": formatted_results, "used_web_search": True}

    except Exception as e:
        print(f"Error in Tavily search: {e}")
        return {"web_search_results": "Tidak ada hasil pencarian web yang ditemukan"}

def generate_final_answer(state: AgentState) -> AgentState:
    question = state["question"]
    docs_answer = state["answer"]
    web_results = state.get("web_search_results", "")

    messages = [
        {"role": "system", "content": "Anda adalah asisten yang membantu memberikan jawaban komprehensif berdasarkan semua informasi yang tersedia. Jawab dalam bahasa Indonesia. Jawab dalam 1 paragraf."},
        {"role": "user", "content": f"Pertanyaan: {question}\n\nJawaban dari dokumen: {docs_answer}\n\nInformasi tambahan dari pencarian web: {web_results}\n\nBerikan jawaban akhir yang lengkap dan akurat:"}
    ]

    final_answer = llm(messages)

    return {"final_answer": final_answer}

def should_search_web(state: AgentState) -> str:
    feedback = state["feedback"]

    is_sufficient = feedback.get("is_sufficient", False)

    if is_sufficient:
        return "sufficient"
    else:
        return "insufficient"

def create_workflow():
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve_documents", retrieve_documents)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("judge_answer", judge_answer)
    workflow.add_node("search_web", search_web)
    workflow.add_node("generate_final_answer", generate_final_answer)

    workflow.add_edge(START, "retrieve_documents")
    workflow.add_edge("retrieve_documents", "generate_answer")
    workflow.add_edge("generate_answer", "judge_answer")
    workflow.add_conditional_edges(
        "judge_answer",
        should_search_web,
        {
            "sufficient": END,
            "insufficient": "search_web"
        }
    )
    workflow.add_edge("search_web", "generate_final_answer")
    workflow.add_edge("generate_final_answer", END)

    return workflow.compile()

def run_agent(question: str) -> str:
    agent_executor = create_workflow()
    result = agent_executor.invoke(
        {"question": question, "used_web_search": False})
    return result

def main():
    df_questions = pd.read_csv("/workspace/questions_with_answers_test_with_filename.csv")
    questions = df_questions['question'].tolist()
    sealion_df = pd.read_csv("sahabatai9b_test.csv")

    try:
        for i, question in enumerate(questions[START_INDEX:]):
            print(f"Answering question ({i+START_INDEX}) {question}: ", end="")
            result = run_agent(question)
            sealion_df.loc[i+START_INDEX] = {
                "question": question,
                "answer": result['final_answer'],
                "filenames": json.dumps(result['filenames']),
                "used_web_search": result['used_web_search']
            }

            print(result['final_answer'])
            sealion_df.to_csv("sahabatai9b_test.csv", index=False)
    except Exception as e:
        print("Error: ", e)

if __name__ == "__main__":
    main()                    


# In[ ]:




