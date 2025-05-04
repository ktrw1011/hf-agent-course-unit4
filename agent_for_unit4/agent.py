import base64
import os
from pathlib import Path
from textwrap import dedent

from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel, VisitWebpageTool

from .tools import RetrieveCSVStorageTool, SpeechRecognitionTool, VisualQATool, WikiTool, fetch_text_content, read_excel


def configure_open_telemetry() -> None:
    try:
        from openinference.instrumentation.smolagents import SmolagentsInstrumentor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    except ImportError:
        print("OpenTelemetry packages are not installed. Please install them to enable tracing.")
        return None

    try:
        langfuse_public_key = os.environ["LANGFUSE_PUBLIC_KEY"]
        langfuse_secret_key = os.environ["LANGFUSE_SECRET_KEY"]
    except KeyError:
        print("LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be set in the environment variables.")
        return None

    LANGFUSE_AUTH = base64.b64encode(f"{langfuse_public_key}:{langfuse_secret_key}".encode()).decode()
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://cloud.langfuse.com/api/public/otel"
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"

    trace_provider = TracerProvider()
    trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))

    SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)


configure_open_telemetry()

wiki_storage_tool = RetrieveCSVStorageTool(
    table_name="wiki",
    init_storage=True,
    storage_path="./storage",
)

wiki_agent = CodeAgent(
    name="wiki_agent",
    description= """A wiki agent that can search and retrieve information from Wikipedia.
    It is specialized for handling wikipedia articles, and is recommended over web_agent for retrieving information from wikipedia.""",
    model=LiteLLMModel(model_id="openrouter/qwen/qwen-2.5-coder-32b-instruct"),
    tools=[
        DuckDuckGoSearchTool(),
        wiki_storage_tool,
        WikiTool(storage=wiki_storage_tool.get_storage()),
    ],
    max_steps=10,
    additional_authorized_imports=["pandas"],
)


web_agent = CodeAgent(
    model=LiteLLMModel(model_id="openrouter/qwen/qwen-2.5-coder-32b-instruct"),
    tools=[
        DuckDuckGoSearchTool(max_results=10),
        VisitWebpageTool(),
    ],
    name="web_agent",
    description="A web agent that can search and visit webpages.",
    verbosity_level=2,
    max_steps=10,
)


manager_agent = CodeAgent(
    model=LiteLLMModel(
        model_id="openrouter/qwen/qwq-32b",
    ),
    tools=[
        fetch_text_content,  # fetch text content from a URL
        SpeechRecognitionTool(),  # Audio to text
        VisualQATool(),  # Visual Question Answering
        read_excel,  # Read Excel files
    ],
    managed_agents=[
        wiki_agent,
        web_agent,
    ],
    additional_authorized_imports=["pandas", "requests"],
    planning_interval=5,
    verbosity_level=2,
    max_steps=15,
)


def parse_file_name(file_base_url: str, file_name: str) -> str:
    if file_name == "":
        return "not provided"
    return file_base_url + Path(file_name).stem


def prepare_for_input(question: dict, file_base_url: str) -> str:
    input_text = dedent(f"""\
    {question["question"]}

    If necessary, use the following file (they may not be provided)
    file_type: {Path(question["file_name"]).suffix}
    file: {parse_file_name(file_base_url, question["file_name"])}

    Video analysis tools are currently unavailable.
    If the question is about analyzing the video (e.g. questions about Youtube link and mp4), answer 'No Answer'.""")
    return input_text
