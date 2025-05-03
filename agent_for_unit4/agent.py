from smolagents import CodeAgent, LiteLLMModel


manager_agent = CodeAgent(
    model=LiteLLMModel(
        model_id="openrouter/qwen/qwq-32B",
    ),
    tools=[],
    managed_agents=[
    ],
    planning_interval=5,
    verbosity_level=2,
    max_steps=15,
)