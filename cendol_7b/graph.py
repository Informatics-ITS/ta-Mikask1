import os
import json
import pandas as pd
from typing import Dict, List, TypedDict, Any
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langgraph.graph import StateGraph, START, END
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from tavily import TavilyClient

# Load environment variables
load_dotenv()

# Constants
DEVICE = "cuda:0"
CHUNK_FOLDER = r"chunks"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
START_INDEX = 0

# Initialize models and clients
tokenizer = AutoTokenizer.from_pretrained("indonlp/cendol-llama2-7b-chat")
model = AutoModelForCausalLM.from_pretrained(
    "indonlp/cendol-llama2-7b-chat").to(DEVICE)
pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer,
                    pad_token_id=tokenizer.eos_token_id, repetition_penalty=1.1, max_new_tokens=1024)

embedding_model = SentenceTransformer(
    'intfloat/multilingual-e5-large-instruct', device="cuda")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("uu-index")
tavily = TavilyClient(api_key=TAVILY_API_KEY)

# State definition
class AgentState(TypedDict):
    question: str
    docs: List[str]
    answer: str
    feedback: Dict[str, Any]
    web_search_results: List[Dict]
    final_answer: str
    filenames: List[str]
    used_web_search: bool

# Utility functions
def llm(prompt):
    return pipeline(prompt)[0]['generated_text']


def cleanup_answer(response: str, answer_key: str) -> str:
    text = response.split(f"{answer_key}:")[-1].strip().replace("[INST]", "").replace(
        "<<SYS>>", "").replace("[INST]", "").replace("<</SYS>>", "").replace("[/SYS]", "").strip()

    text = text.strip()
    last_period_index = text.rfind(".")
    if last_period_index != -1:
        text = text[:last_period_index + 1]

    return f"{answer_key}: " + text


def format_prompt(system: str, instruction: str, answer_key: str) -> str:
    return f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n" + instruction + " [/INST]\n" + f"{answer_key}:\n"

# Node functions
def retrieve_documents(state: AgentState) -> AgentState:
    query = state["question"]
    query_vector = embedding_model.encode(query).tolist()

    query_results = index.query(
        vector=query_vector,
        top_k=5,
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

    context = "\n\n".join(docs)

    prompt = format_prompt(
        system=f"Anda adalah asisten yang membantu menjawab pertanyaan menggunakan konteks yang disediakan. Jika konteks tidak berisi informasi yang diperlukan, katakan dengan jelas.\nKonteks: {context}", instruction=f"Pertanyaan: {question}. Jawab dalam Bahasa Indonesia. Jawab dalam 1 paragraf.", answer_key="Jawaban")

    answer = cleanup_answer(llm(prompt), "Jawaban")

    return {"answer": answer}


def judge_answer(state: AgentState) -> AgentState:
    question = state["question"]
    answer = state["answer"]

    prompt = format_prompt(system=f"Anda adalah seorang hakim yang mengevaluasi apakah jawaban sepenuhnya menjawab pertanyaan.",
                           instruction=f"Pertanyaan: {question}\n\nJawaban: {answer}. Evaluasi jika jawaban ini cukup untuk menjawab pertanyaan. Anda harus mengembalikan \"CUKUP=FALSE\" jika jawaban ini belum cukup.", answer_key="Evaluasi")

    feedback_text = cleanup_answer(llm(prompt), "Evaluasi")

    if "CUKUP=FALSE" in feedback_text:
        return {"feedback": {
            "is_sufficient": False
        }, "final_answer": state['answer']}

    return {"feedback": {
            "is_sufficient": True
            }, "final_answer": state['answer']}


def search_web(state: AgentState) -> AgentState:
    """Perform a web search using Tavily API to find additional information."""
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

    prompt = format_prompt(system=f"Anda adalah asisten yang membantu memberikan jawaban komprehensif berdasarkan semua informasi yang tersedia.\nJawaban dari dokumen: {docs_answer}\n\nInformasi tambahan dari pencarian web: {web_results}\n\n",
                           instruction=f"Pertanyaan: {question}\n\nBerikan jawaban akhir yang lengkap dan akurat. Jawab dalam bahasa Indonesia. Jawab dalam 1 paragraf.", answer_key="Jawaban")

    answer = cleanup_answer(llm(prompt), "Jawaban")

    return {"final_answer": answer.split("Jawaban: ")[1].replace("\n", " ")}


def should_search_web(state: AgentState) -> str:
    feedback = state["feedback"]

    is_sufficient = feedback.get("is_sufficient", False)

    if is_sufficient:
        return "sufficient"
    else:
        return "insufficient"

# Workflow setup


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
            "sufficient": "generate_final_answer",
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

# Main execution


def main():
    # Load data
    df_questions = pd.read_csv(
        "/workspace/questions_with_answers_test_with_filename.csv")
    questions = df_questions['question'].tolist()
    cendol_df = pd.read_csv("cendol7b_test.csv")

    try:
        for i, question in enumerate(questions[START_INDEX:]):
            print(f"Question ({i+START_INDEX}) {question}: ", end="")
            result = run_agent(question)
            cendol_df.loc[i+START_INDEX] = {
                "question": question,
                "answer": result['final_answer'],
                "filenames": json.dumps(result['filenames']),
                "used_web_search": result['used_web_search']
            }

            cendol_df.to_csv("cendol7b_test.csv", index=False)
            print(result['final_answer'], end="\n\n")
    except Exception as e:
        print("Error: ", e)


if __name__ == "__main__":
    main()
