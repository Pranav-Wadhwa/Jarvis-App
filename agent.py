from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, RunContext, function_tool, inference 
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from zoneinfo import ZoneInfo

import logging
from langchain_community.document_loaders import DirectoryLoader
import json

logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("unstructured").setLevel(logging.ERROR)
logging.getLogger("langchain_community").setLevel(logging.ERROR)


load_dotenv(".env.local")

_doc_reader = None


def get_doc_reader():
    global _doc_reader
    if _doc_reader is None:
        _doc_reader = DocReader()
    return _doc_reader


class DocReader:
    def __init__(self):
        loader = DirectoryLoader("docs", glob="**/*")
        documents = loader.load()
        self.all_docs_content = "\n".join([d.page_content for d in documents])

    def get_context(self) -> str:
        print("\n\n--- DOCUMENT CONTEXT ---")
        print(self.all_docs_content)
        print("---\n\n")
        return self.all_docs_content


def load_instructions(filename: str) -> str:
    return Path(__file__).with_name(filename).read_text(encoding="utf-8").strip()


class Assistant(Agent):
    def __init__(self, **kwargs: Any) -> None:
        with open("agents.json", "r") as f:
            agents_config = json.load(f)
        
        agent_names = list(agents_config.keys())
        agent_list_str = "\n".join([f"- {name}" for name in agent_names])
        agent_names_str = ", ".join(agent_names)

        instructions = load_instructions("launcher-instructions.txt").format(
            agent_list=agent_list_str,
            agent_names=agent_names_str,
        )

        super().__init__(instructions=instructions, **kwargs)

    @function_tool()
    async def handoff(self, context: RunContext, agent_name: str):
        """Handoff the conversation to another agent."""
        if agent_name == "Builder":
            return BuilderAgent(chat_ctx=self.chat_ctx), "Ok, handing off to the Builder."

        with open("agents.json", "r") as f:
            agents_config = json.load(f)

        if agent_name in agents_config:
            return (
                ConfigurableAgent(agent_name=agent_name, chat_ctx=self.chat_ctx),
                f"Ok, handing off to {agent_name}.",
            )

        return None, f"I'm sorry, I don't know an agent named {agent_name}."


class BuilderAgent(Agent):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(instructions=load_instructions("builder.txt"), **kwargs)
        self.doc_reader = get_doc_reader()

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions=(
                "Greet the user as the Builder agent and ask what kind of voice app "
                "they would like to create."
            )
        )

    @function_tool()
    async def get_context(self, context: RunContext):
        """Get the full content of all documents in the /docs folder."""
        answer = self.doc_reader.get_context()
        return None, answer

    @function_tool()
    async def create_new_agent(self, context: RunContext, name: str):
        """Create a new agent based on the name provided."""
        # In a real application, you would instantiate the new agent and add it to the session
        # For now, we'll just return a success message.
        return None, f"Successfully created the {name} agent."


class ConfigurableAgent(Agent):
    def __init__(self, agent_name: str, **kwargs: Any) -> None:
        with open("agents.json", "r") as f:
            agents_config = json.load(f)
        
        config = agents_config[agent_name]
        super().__init__(instructions=config["instructions"], **kwargs)

        self.doc_reader = get_doc_reader()

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
    async def get_context(self, context: RunContext):
        """Get the full content of all documents in the /docs folder."""
        answer = self.doc_reader.get_context()
        return None, answer


class GetTimeAgent(Agent):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(instructions=load_instructions("time-agent-instructions.txt"), **kwargs)

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

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions=(
                "Greet the user, explain you are the GetTime agent, and mention "
                "you report the current date and time using Pacific time unless they "
                "ask for another timezone."
            )
        )



async def entrypoint(ctx: agents.JobContext):

    llm = inference.LLM(model="openai/gpt-4.1", provider="azure")
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