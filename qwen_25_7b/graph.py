import os
import re
import json
import pandas as pd
from typing import Dict, List, TypedDict, Any, Optional, Mapping
from abc import ABC
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import LLMResult, ChatGeneration
from langchain.schema import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from tavily import TavilyClient

load_dotenv()

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
CHUNK_FOLDER = r"chunks"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
START_INDEX = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map=DEVICE
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

embedding_model = SentenceTransformer(
    'intfloat/multilingual-e5-large-instruct', device=DEVICE)
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


class Qwen(LLM, ABC):
    max_token: int = 10000
    temperature: float = 0.01
    history_len: int = 3

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "Qwen"

    @property
    def _history_len(self) -> int:
        return self.history_len

    def set_history_len(self, history_len: int = 10) -> None:
        self.history_len = history_len

    def _convert_messages(self, conversation: List[BaseMessage]):
        role_map = {
            SystemMessage: "system",
            HumanMessage: "user",
            AIMessage: "assistant"
        }

        messages = []
        for message in conversation:
            role = role_map.get(type(message))
            if role:
                messages.append({
                    "role": role,
                    "content": message.content
                })
        return messages

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True)[0]
        return response

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        message_dicts = self._convert_messages(messages)
        text = tokenizer.apply_chat_template(
            message_dicts,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True)[0]
        generations = [ChatGeneration(
            message=AIMessage(response)
        )]

        return LLMResult(generations=[generations])

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"max_token": self.max_token,
                "temperature": self.temperature,
                "history_len": self.history_len}


llm = Qwen()
judge_llm = Qwen()


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
        top_k=12,
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

    messages = [
        SystemMessage(content="Anda adalah asisten yang membantu menjawab pertanyaan menggunakan konteks yang disediakan. Jika konteks tidak berisi informasi yang diperlukan, katakan dengan jelas. Jawab dalam Bahasa Indonesia. Jawab dalam 1 paragraf."),
        HumanMessage(
            content=f"Konteks: {context}\n\nPertanyaan: {question}\n\nJawaban:")
    ]

    answer = cleanup_answer(llm.generate(messages).generations[0][0].text)

    return {"answer": answer}


def judge_answer(state: AgentState) -> AgentState:
    question = state["question"]
    answer = state["answer"]

    messages = [
        SystemMessage(content="Anda adalah seorang hakim yang mengevaluasi apakah jawaban sepenuhnya menjawab pertanyaan. PENTING: Anda harus mengembalikan \"IS_SUFFICIENT=TRUE\" jika jawaban ini cukup."),
        HumanMessage(
            content=f"Pertanyaan: {question}\n\nJawaban: {answer}\n\nEvaluasi apakah jawaban ini cukup:")
    ]

    feedback_text = judge_llm.generate(messages).generations[0][0].text

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
        SystemMessage(content="Anda adalah asisten yang membantu memberikan jawaban komprehensif berdasarkan semua informasi yang tersedia. Jawab dalam bahasa Indonesia. Jawab dalam 1 paragraf."),
        HumanMessage(
            content=f"Pertanyaan: {question}\n\nJawaban dari dokumen: {docs_answer}\n\nInformasi tambahan dari pencarian web: {web_results}\n\nBerikan jawaban akhir yang lengkap dan akurat:")
    ]

    final_answer = cleanup_answer(
        llm.generate(messages).generations[0][0].text)

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
    df_questions = pd.read_csv(
        "/workspace/questions_with_answers_test_with_filename.csv")
    questions = df_questions['question'].tolist()
    qwen_df = pd.read_csv("qwen7b_test.csv")

    try:
        for i, question in enumerate(questions[START_INDEX:]):
            print(f"Answering question ({i+START_INDEX}) {question}: ", end="")
            result = run_agent(question)
            qwen_df.loc[i+START_INDEX] = {
                "question": question,
                "answer": result['final_answer'],
                "filenames": json.dumps(result['filenames']),
                "used_web_search": result['used_web_search']
            }

            qwen_df.to_csv("qwen7b_test.csv", index=False)
            print(result['final_answer'])
    except Exception as e:
        print("Error: ", e)


if __name__ == "__main__":
    main()
