from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, RunContext, function_tool, inference 
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from zoneinfo import ZoneInfo
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
import json
import logging

# Suppress noisy loggers
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("unstructured").setLevel(logging.ERROR)
logging.getLogger("langchain_community").setLevel(logging.ERROR)


load_dotenv(".env.local")

_doc_retriever = None


def get_doc_retriever():
    global _doc_retriever
    if _doc_retriever is None:
        _doc_retriever = DocRetriever()
    return _doc_retriever


class DocRetriever:
    def __init__(self):
        loader = DirectoryLoader("docs", glob="**/*")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents(docs, embeddings)
        self.retriever = db.as_retriever()

    def query(self, question: str) -> str:
        print("\n\n--- RAG DIAGNOSTICS ---")
        print(f"RAG Question from Agent: {question}")
        docs = self.retriever.invoke(question)
        if not docs:
            print("RAG Context: No relevant documents found.")
            print("---\n\n")
            return "I couldn't find any relevant information in the documents."

        context_str = "\n".join([d.page_content for d in docs])
        print(f"RAG Context Sent to LLM:\n---\n{context_str}\n---")
        print("---\n\n")
        return context_str


def load_instructions(filename: str) -> str:
    return Path(__file__).with_name(filename).read_text(encoding="utf-8").strip()


class Assistant(Agent):
    def __init__(self, **kwargs: Any) -> None:
        instructions_template = load_instructions("launcher-instructions.txt")

        with open("agents.json", "r") as f:
            agents_config = json.load(f)

        static_agents = {
            "Builder": "Use this agent to create a new voice app.",
            "RAG": "Use this agent to answer questions about documents.",
        }

        all_agents = {**static_agents, **{name: cfg["instructions"] for name, cfg in agents_config.items()}}
        
        agent_list_txt = "\n".join([f"- {name}: {desc}" for name, desc in all_agents.items()])
        agent_names_txt = ", ".join(all_agents.keys())

        instructions = instructions_template.format(
            agent_list=agent_list_txt, agent_names=agent_names_txt
        )

        super().__init__(instructions=instructions, **kwargs)

    @function_tool()
    async def handoff(self, context: RunContext, agent_name: str):
        """Hand off the conversation to another agent by its name."""
        if agent_name == "Builder":
            return BuilderAgent(chat_ctx=self.chat_ctx), f"Ok, handing off to the {agent_name} agent."
        
        if agent_name == "RAG":
            return RAGAgent(chat_ctx=self.chat_ctx), f"Ok, handing off to the {agent_name} agent."
        
        with open("agents.json", "r") as f:
            agents_config = json.load(f)
        
        if agent_name in agents_config:
            return ConfigurableAgent(agent_name=agent_name, chat_ctx=self.chat_ctx), f"Ok, handing off to the {agent_name} agent."
        
        return None, f"I'm sorry, I don't know an agent named {agent_name}."


class BuilderAgent(Agent):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(instructions=load_instructions("builder.txt"), **kwargs)
        self.doc_retriever = get_doc_retriever()

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions=(
                "Greet the user as the Builder agent and ask what kind of voice app "
                "they would like to create."
            )
        )

    @function_tool()
    async def search_documentation(self, context: RunContext, query: str):
        """Search the documentation for an answer to a question that will help you build a new voice app."""
        answer = self.doc_retriever.query(query)
        return None, answer

    @function_tool()
    async def create_new_agent(
        self,
        context: RunContext,
        name: str,
        instructions: str,
        tools: list[str] | None = None,
    ):
        """Create a new voice agent with the given name, instructions, and tools.
        The `tools` argument must be a list containing either 'search_documentation' or 'get_current_time' or both."""
        with open("agents.json", "r+") as f:
            agents_config = json.load(f)
            agents_config[name] = {
                "instructions": instructions,
                "tools": tools or [],
            }
            f.seek(0)
            json.dump(agents_config, f, indent=4)
            f.truncate()

        return None, f"Successfully created the {name} agent."


class RAGAgent(Agent):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            instructions="You are a helpful assistant that can answer questions about documents.",
            **kwargs,
        )
        self.doc_retriever = get_doc_retriever()

    @function_tool()
    async def answer_question(self, context: RunContext, question: str):
        """Answer a question about the documents."""
        answer = self.doc_retriever.query(question)
        return None, answer


class ConfigurableAgent(Agent):
    def __init__(self, agent_name: str, **kwargs: Any) -> None:
        with open("agents.json", "r") as f:
            agents_config = json.load(f)
        
        config = agents_config[agent_name]
        super().__init__(instructions=config["instructions"], **kwargs)

        self.doc_retriever = get_doc_retriever()

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Greet the user and begin performing your main function based on your instructions."
        )

    @function_tool()
    async def get_current_time(self, context: RunContext, timezone: str | None = None):
        """Return the current date and time in the requested timezone, defaulting to US Pacific."""
        tz_name = timezone or "America/Los_Angeles"
        try:
            tz = ZoneInfo(tz_name)
        except Exception as exc:  # pragma: no cover - invalid zone names handled gracefully
            return (
                None,
                f"I couldn't interpret the timezone '{tz_name}'. "
                "Please provide an IANA timezone identifier like 'America/New_York'.",
            )

        now = datetime.now(tz)
        timestamp = now.strftime("%A, %B %d, %Y %I:%M %p %Z")
        return None, f"The current date and time is {timestamp}."

    @function_tool()
    async def search_documentation(self, context: RunContext, query: str):
        """Search the documentation for an answer to a question."""
        answer = self.doc_retriever.query(query)
        return None, answer


async def entrypoint(ctx: agents.JobContext):

    llm = inference.LLM(model="openai/gpt-4.1")
    #  llm = inference.LLM(model="openai/gpt-5-mini", provider="azure", extra_kwargs={"reasoning_effort": "minimal"})
    tts = inference.TTS(model="rime/mistv2", voice="geoff")

    session = AgentSession(
        stt="assemblyai/universal-streaming:en",
        llm=llm,
        tts=tts,
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` instead for best results
            noise_cancellation=noise_cancellation.BVC(), 
        ),
    )

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))