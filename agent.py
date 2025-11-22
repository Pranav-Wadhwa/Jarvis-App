import csv
import io
import json
from collections import namedtuple
from datetime import datetime
from pathlib import Path
from typing import Any
import base64
import os
import asyncio

try:
    import requests
except ImportError:
    requests = None

from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredFileLoader
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, RunContext, function_tool, inference, metrics, MetricsCollectedEvent, ChatContext, ChatMessage, BackgroundAudioPlayer, AudioConfig, BuiltinAudioClip
from livekit.plugins import noise_cancellation, silero, deepgram, rime, openai
from livekit.agents.telemetry import set_tracer_provider
from livekit.plugins.turn_detector.multilingual import MultilingualModel




load_dotenv(".env.local")

App = namedtuple("App", ["id", "name", "description"])

def setup_langfuse(
    host: str | None = None, public_key: str | None = None, secret_key: str | None = None
):
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
    host = host or os.getenv("LANGFUSE_HOST")

    if not public_key or not secret_key or not host:
        raise ValueError("LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_HOST must be set")

    langfuse_auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"{host.rstrip('/')}/api/public/otel"
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {langfuse_auth}"

    trace_provider = TracerProvider()
    trace_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    set_tracer_provider(trace_provider)


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
                    description=row.get("description", "")
                ))
    
    return apps


def load_memory(file_path: str) -> list[dict[str, str]]:
    """Load memory from a CSV file relative to ./data directory.
    
    Args:
        file_path: Path relative to ./data (e.g., "app_memories/app_id.csv" or "shared_memory.csv")
    
    Returns:
        A list of dictionaries with keys: id, timestamp, content.
        Returns an empty list if the file doesn't exist.
    """
    memory_path = Path(__file__).with_name("data") / file_path
    
    if not memory_path.exists():
        return []
    
    memories = []
    with open(memory_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("id"):  # Skip empty rows
                memories.append({
                    "id": row["id"],
                    "timestamp": row.get("timestamp", ""),
                    "content": row.get("content", "")
                })
    
    return memories


def format_memories_as_csv(memories: list[dict[str, str]]) -> str:
    """Format memories as CSV string.
    
    Args:
        memories: List of dictionaries with keys: id, timestamp, content
    
    Returns:
        CSV formatted string with header row
    """
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["id", "timestamp", "content"])
    writer.writeheader()
    writer.writerows(memories)
    return output.getvalue()


def write_memory_csv(file_path: str, memories: list[dict[str, str]]) -> None:
    """Write memories to a CSV file relative to ./data directory.
    
    Args:
        file_path: Path relative to ./data (e.g., "app_memories/app_id.csv" or "shared_memory.csv")
        memories: List of dictionaries with keys: id, timestamp, content
    """
    memory_path = Path(__file__).with_name("data") / file_path
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(memory_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "timestamp", "content"])
        writer.writeheader()
        writer.writerows(memories)


_docs_content = None

def get_docs_content() -> str:
    """Load docs from the global /docs directory and cache the content."""
    global _docs_content
    if _docs_content is not None:
        return _docs_content

    docs_path = Path(__file__).parent / "docs"
    
    if not docs_path.exists() or not docs_path.is_dir():
        _docs_content = ""
        return _docs_content
    
    try:
        loader = DirectoryLoader(str(docs_path), glob="**/*")
        documents = loader.load()
        _docs_content = "\n".join([d.page_content for d in documents])
        return _docs_content
    except Exception as e:
        print(f"Error loading documents from /docs: {e}")
        _docs_content = ""
        return _docs_content


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
        
        # Load and append shared memory
        shared_memories = load_memory("shared_memory.csv")
        if shared_memories:
            csv_data = format_memories_as_csv(shared_memories)
            instructions += f"\n\n<shared_memory>\n{csv_data}</shared_memory>"
        
        super().__init__(instructions=instructions, **kwargs)

# this is a background task that says "mmhmm" every time the user completes a turn
#    async def _say_background(self):
#        self.session.say("Sure thing.", allow_interruptions=True)

#    async def on_user_turn_completed(
#        self, turn_ctx: RunContext, new_message: ChatMessage) -> None:
#        asyncio.create_task(self._say_background())


    @function_tool()
    async def handoff_to_builder(self, context: RunContext):
        """Hand off the conversation to the Builder agent when the user wants to create a new app."""
        print("Handing off to Builder agent")
        return BuilderAgent(chat_ctx=self.chat_ctx), "Ok."

    @function_tool()
    async def handoff_to_app(self, context: RunContext, app_id: str):
        """Hand off the conversation to the specified app agent."""
        return AppAgent(app_id=app_id, chat_ctx=self.chat_ctx), "Ok."

    @function_tool()
    async def add_to_shared_memory(self, context: RunContext, id: str, value: str):
        """Adds a new memory entry to the shared memory storage that is accessible across all applications in the agentic voice system. Shared memory is designed for storing generic, user-level information that should be available to any application, such as the user's name, preferred language, timezone, or other universal preferences. Unlike app-specific memory, shared memory entries can be accessed and used by any application launched within the system, enabling a consistent user experience across different voice applications. This is particularly useful for information that doesn't change frequently and should persist across different application contexts."""
        memories = load_memory("shared_memory.csv")
        
        # Check if id already exists
        for memory in memories:
            if memory["id"] == id:
                return None, f"Shared memory entry with id '{id}' already exists."
        
        # Add new memory entry with current timestamp
        timestamp = datetime.now().isoformat()
        memories.append({"id": id, "timestamp": timestamp, "content": value})
        write_memory_csv("shared_memory.csv", memories)
        
        return None, f"Successfully added shared memory entry '{id}'."


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
        memories_dir = data_dir / "app_memories"
        memory_path = memories_dir / f"{app_id}.csv"
        
        # Create data directory and subdirectories if they don't exist
        data_dir.mkdir(exist_ok=True)
        instructions_dir.mkdir(exist_ok=True)
        memories_dir.mkdir(exist_ok=True)
        
        # Write system prompt file
        instructions_path.write_text(system_prompt, encoding="utf-8")
        
        # Create memory CSV file with headers if it doesn't exist
        if not memory_path.exists():
            with open(memory_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["id", "timestamp", "content"])
                writer.writeheader()
        
        # Append to CSV
        file_exists = csv_path.exists()
        with open(csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "name", "description"])
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "id": app_id,
                "name": name,
                "description": description,
            })
        
        # Return the AppAgent so it becomes active
        return AppAgent(app_id=app_id), f"Successfully created app '{name}'."

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions=(
                "Greet the user as the Builder agent and ask what kind of voice app "
                "they would like to create."
            )
        )


class AppAgent(Agent):
    def __init__(self, app_id: str, **kwargs: Any) -> None:
        self.app_id = app_id
        
        # Load app agent instructions (common to all apps)
        app_agent_instructions = load_instructions("app_agent_instructions.txt")
        
        # Load system prompt from data/app_system_instructions/<app_id>.txt
        instructions_path = Path(__file__).with_name("data") / "app_system_instructions" / f"{app_id}.txt"
        if instructions_path.exists():
            app_specific_instructions = instructions_path.read_text(encoding="utf-8").strip()
        else:
            app_specific_instructions = f"System instructions for app {app_id} not found."
        
        # Combine instructions: app_agent_instructions first, then app-specific instructions
        instructions = f"{app_agent_instructions}\n\n{app_specific_instructions}" if app_agent_instructions.strip() else app_specific_instructions
        
        # Load and append shared memory (above app memory)
        shared_memories = load_memory("shared_memory.csv")
        if shared_memories:
            csv_data = format_memories_as_csv(shared_memories)
            instructions += f"\n\n<shared_memory>\n{csv_data}</shared_memory>"
        
        # Load and append app memories
        memories = load_memory(f"app_memories/{app_id}.csv")
        if memories:
            csv_data = format_memories_as_csv(memories)
            instructions += f"\n\n<app_memory>\n{csv_data}</app_memory>"
        
        super().__init__(instructions=instructions, **kwargs)

    @function_tool()
    async def list_available_docs(self, context: RunContext):
        """List the filenames of all documents available in the /docs directory."""
        docs_path = Path(__file__).parent / "docs"
        if not docs_path.exists() or not docs_path.is_dir():
            return None, "There are no documents available."
        
        files = [f.name for f in docs_path.iterdir() if f.is_file()]
        if not files:
            return None, "There are no documents available."
        
        return None, "The following documents are available:\n" + "\n".join(files)

    @function_tool()
    async def get_doc_by_filename(self, context: RunContext, filename: str):
        """Read the content of a specific file from the /docs directory."""
        docs_path = Path(__file__).parent / "docs"
        file_path = docs_path / filename

        if not file_path.exists() or not file_path.is_file():
            return None, f"The file '{filename}' was not found."

        try:
            loader = UnstructuredFileLoader(str(file_path))
            docs = loader.load()
            return None, "\n".join([doc.page_content for doc in docs])
        except Exception as e:
            return None, f"Error reading file '{filename}': {e}"

    @function_tool()
    async def add_to_app_memory(self, context: RunContext, id: str, value: str):
        """Adds a new memory entry to this application's persistent memory storage. This allows the voice application to store and recall information across conversations and sessions. The memory system enables the application to maintain context, remember user preferences, store conversation history, and persist any data that should be available in future interactions. Each memory entry is uniquely identified by an id, allowing the application to retrieve, update, or delete specific memories later."""
        memories = load_memory(f"app_memories/{self.app_id}.csv")
        
        # Check if id already exists
        for memory in memories:
            if memory["id"] == id:
                return None, f"Memory entry with id '{id}' already exists. Use update_app_memory to modify it."
        
        # Add new memory entry with current timestamp
        timestamp = datetime.now().isoformat()
        memories.append({"id": id, "timestamp": timestamp, "content": value})
        write_memory_csv(f"app_memories/{self.app_id}.csv", memories)
        
        return None, f"Successfully added memory entry '{id}'."

    @function_tool()
    async def add_to_shared_memory(self, context: RunContext, id: str, value: str):
        """Adds a new memory entry to the shared memory storage that is accessible across all applications in the agentic voice system. Shared memory is designed for storing generic, user-level information that should be available to any application, such as the user's name, preferred language, timezone, or other universal preferences. Unlike app-specific memory, shared memory entries can be accessed and used by any application launched within the system, enabling a consistent user experience across different voice applications. This is particularly useful for information that doesn't change frequently and should persist across different application contexts."""
        memories = load_memory("shared_memory.csv")
        
        # Check if id already exists
        for memory in memories:
            if memory["id"] == id:
                return None, f"Shared memory entry with id '{id}' already exists."
        
        # Add new memory entry with current timestamp
        timestamp = datetime.now().isoformat()
        memories.append({"id": id, "timestamp": timestamp, "content": value})
        write_memory_csv("shared_memory.csv", memories)
        
        return None, f"Successfully added shared memory entry '{id}'."

    @function_tool()
    async def update_app_memory(self, context: RunContext, id: str, new_value: str):
        """Updates an existing memory entry in this application's persistent memory storage. This allows the voice application to modify previously stored information when circumstances change or when new information becomes available. The memory entry must already exist (created via add_to_app_memory) for this operation to succeed. This is useful for updating user preferences, correcting stored information, or refreshing conversation context as the interaction progresses."""
        memories = load_memory(f"app_memories/{self.app_id}.csv")
        
        # Find and update the memory entry with current timestamp
        found = False
        timestamp = datetime.now().isoformat()
        for memory in memories:
            if memory["id"] == id:
                memory["content"] = new_value
                memory["timestamp"] = timestamp
                found = True
                break
        
        if not found:
            return None, f"Memory entry with id '{id}' not found. Use add_to_app_memory to create it."
        
        write_memory_csv(f"app_memories/{self.app_id}.csv", memories)
        return None, f"Successfully updated memory entry '{id}'."

    @function_tool()
    async def delete_app_memory(self, context: RunContext, id: str):
        """Deletes an existing memory entry from this application's persistent memory storage. This allows the voice application to remove information that is no longer needed, incorrect, or should be forgotten. Once deleted, the memory entry cannot be retrieved, and any future references to this id will not find the previously stored information. This is useful for cleaning up outdated information, removing sensitive data, or resetting specific aspects of the application's memory."""
        memories = load_memory(f"app_memories/{self.app_id}.csv")
        
        # Find and remove the memory entry
        original_count = len(memories)
        memories = [m for m in memories if m["id"] != id]
        
        if len(memories) == original_count:
            return None, f"Memory entry with id '{id}' not found."
        
        write_memory_csv(f"app_memories/{self.app_id}.csv", memories)
        return None, f"Successfully deleted memory entry '{id}'."

    @function_tool()
    async def web_search(self, context: RunContext, query: str):
        """Perform a web search to find current information, facts, news, or any online content. Use this tool when you need to look up information that may not be in your training data, check current events, verify facts, or find specific details about topics, people, places, or things. This is particularly useful for real-time information, recent news, current prices, weather updates, or any information that changes frequently."""
        if requests is None:
            return None, "Web search is not available. Please install requests: pip install requests"
        
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            return None, "SERPER_API_KEY environment variable is not set. Please set it to use web search."
        
        try:
            url = "https://google.serper.dev/search"
            headers = {
                "X-API-KEY": api_key,
                "Content-Type": "application/json"
            }
            payload = {
                "q": query,
                "num": 5
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            organic_results = data.get("organic", [])
            
            if not organic_results:
                return None, f"No results found for query: {query}"
            
            # Format results
            formatted_results = []
            for i, result in enumerate(organic_results, 1):
                formatted_results.append(
                    f"{i}. Title: {result.get('title', 'N/A')}\n"
                    f"   URL: {result.get('link', 'N/A')}\n"
                    f"   Snippet: {result.get('snippet', 'N/A')}\n"
                )
            
            result_text = "\n".join(formatted_results)
            return None, f"Web search results for '{query}':\n\n{result_text}"
        
        except requests.exceptions.RequestException as e:
            return None, f"Error performing web search: {str(e)}"
        except Exception as e:
            return None, f"Unexpected error during web search: {str(e)}"
    
    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Greet the user."
        )



async def entrypoint(ctx: agents.JobContext):

#    llm = inference.LLM(model="openai/gpt-4.1", provider="azure")
    llm = inference.LLM(model="openai/gpt-oss-120b", provider="baseten")
    #  llm = inference.LLM(model="openai/gpt-5-mini", provider="azure", extra_kwargs={"reasoning_effort": "minimal"})
 #   tts = inference.TTS(model="cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc")
        # Create TTS instance with Voxy's voice
    tts = rime.TTS(model="mistv2", speaker="geoff")
#    llm=openai.realtime.RealtimeModel(modalities=["text"])  # OpenAI Realtime Model for text-only interactions
    stt_model=deepgram.STTv2(
        model="flux-general-en",
        eager_eot_threshold=0.4,  # For low-latency responses 0.5 is default
        eot_threshold=0.8,        # Standard turn detection
        eot_timeout_ms=4000,      # Maximum wait time
        sample_rate=16000,        # Audio sample rate
#        keyterms=["specific", "terms"],  # Optional: improve recognition
        )

    session = AgentSession(
        stt=stt_model,
        llm=llm,
        tts=tts,
        turn_detection="stt"
    )

# Log metrics
    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)

    setup_langfuse()

# start session
    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` instead for best results
            noise_cancellation=noise_cancellation.BVC(), 
        ),
    )

    # An audio player with automated ambient and thinking sounds
    background_audio = BackgroundAudioPlayer(
        ambient_sound=AudioConfig(BuiltinAudioClip.OFFICE_AMBIENCE, volume=0.8),
        thinking_sound=[
            AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=0.8),
            AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING2, volume=0.7),
        ],
    )

    await background_audio.start(room=ctx.room, agent_session=session)

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="jarvis")
    )