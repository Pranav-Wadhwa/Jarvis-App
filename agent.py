import json
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, RunContext, function_tool, inference 
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from zoneinfo import ZoneInfo

load_dotenv(".env.local")


def load_instructions(filename: str) -> str:
    return Path(__file__).with_name(filename).read_text(encoding="utf-8").strip()

def load_database(filename: str) -> dict[str, Any]:
    """Load a JSON database file from the same directory as this script."""
    file_path = Path(__file__).with_name(filename)
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f) 

class Assistant(Agent):
    def __init__(self, **kwargs: Any) -> None:
        # Load base instructions
        base_instructions = load_instructions("launcher-instructions.txt")
        
        # Load apps database and format it
        apps_db = load_database("apps-database.json")
        apps_list = "\n".join(
            f"- {app['name']}: {app['description']}"
            for app in apps_db["apps"]
        )
        
        # Combine instructions with apps database
        full_instructions = f"{base_instructions}\n\nAvailable apps:\n{apps_list}"
        
        super().__init__(
            instructions=full_instructions, **kwargs
        )

    @function_tool()
    async def handoff_to_builder(self, context: RunContext):
        """Hand off the conversation to the Builder agent when the user wants to create a new app."""
        return BuilderAgent(chat_ctx=self.chat_ctx), "Ok."

    @function_tool()
    async def handoff_to_time_agent(self, context: RunContext):
        """Hand off the conversation to the GetTime agent when the user asks for the current time."""
        return GetTimeAgent(chat_ctx=self.chat_ctx), "Ok."


class BuilderAgent(Agent):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(instructions=load_instructions("builder.txt"), **kwargs)

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions=(
                "Greet the user as the Builder agent and ask what kind of voice app "
                "they would like to create."
            )
        )


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