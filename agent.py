import csv
import json
from collections import namedtuple
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, RunContext, function_tool, inference 
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(".env.local")

App = namedtuple("App", ["id", "name", "description", "system_prompt"])


def load_instructions(filename: str) -> str:
    return Path(__file__).with_name(filename).read_text(encoding="utf-8").strip()


def load_apps() -> list[App]:
    """Load apps from app_definitions.csv and return a list of App namedtuples."""
    csv_path = Path(__file__).with_name("data") / "app_definitions.csv"
    apps = []
    
    if not csv_path.exists():
        return apps
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("id") and row.get("name"):  # Skip empty rows
                apps.append(App(
                    id=row["id"],
                    name=row["name"],
                    description=row.get("description", ""),
                    system_prompt=row.get("system_prompt", "")
                ))
    
    return apps


class Assistant(Agent):
    def __init__(self, **kwargs: Any) -> None:
        instructions = load_instructions("launcher-instructions.txt")
        apps = load_apps()
        
        # Format apps as JSON
        apps_json = json.dumps([app._asdict() for app in apps], indent=2)
        
        # Insert apps into instructions
        instructions = instructions.replace(
            "<available_apps></available_apps>",
            f"<available_apps>\n{apps_json}\n</available_apps>"
        )
        
        super().__init__(instructions=instructions, **kwargs)

    @function_tool()
    async def handoff_to_builder(self, context: RunContext):
        """Hand off the conversation to the Builder agent when the user wants to create a new app."""
        print("Handing off to Builder agent")
        return BuilderAgent(chat_ctx=self.chat_ctx), "Ok."

    @function_tool()
    async def handoff_to_app(self, context: RunContext, app_id: str):
        """Hand off the conversation to the specified app agent."""
        return AppAgent(app_id=app_id, chat_ctx=self.chat_ctx), "Ok."


class BuilderAgent(Agent):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(instructions=load_instructions("builder.txt"), **kwargs)

    @function_tool()
    async def create_app(
        self,
        context: RunContext,
        app_id: str,
        name: str,
        description: str,
        system_prompt: str,
    ):
        """Create a new app by adding it to app_definitions.csv and creating its system prompt file."""
        data_dir = Path(__file__).with_name("data")
        csv_path = data_dir / "app_definitions.csv"
        instructions_dir = data_dir / "app_system_instructions"
        instructions_path = instructions_dir / f"{app_id}.txt"
        
        # Create data directory and app_system_instructions directory if they don't exist
        data_dir.mkdir(exist_ok=True)
        instructions_dir.mkdir(exist_ok=True)
        
        # Write system prompt file
        instructions_path.write_text(system_prompt, encoding="utf-8")
        
        # Append to CSV
        file_exists = csv_path.exists()
        with open(csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "name", "description", "system_prompt"])
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "id": app_id,
                "name": name,
                "description": description,
                "system_prompt": system_prompt,
            })
        
        # Return the AppAgent so it becomes active
        return AppAgent(app_id=app_id, chat_ctx=self.chat_ctx), f"Successfully created app '{name}'."

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions=(
                "Greet the user as the Builder agent and ask what kind of voice app "
                "they would like to create."
            )
        )


class AppAgent(Agent):
    def __init__(self, app_id: str, **kwargs: Any) -> None:
        # Load system prompt from data/app_system_instructions/<app_id>.txt
        instructions_path = Path(__file__).with_name("data") / "app_system_instructions" / f"{app_id}.txt"
        if instructions_path.exists():
            instructions = instructions_path.read_text(encoding="utf-8").strip()
        else:
            instructions = f"System instructions for app {app_id} not found."
        
        super().__init__(instructions=instructions, **kwargs)

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Greet the user."
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