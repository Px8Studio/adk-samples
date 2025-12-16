# Using ADK Agent with τ-bench

This guide provides instructions on how to set up and use a custom agent that integrates with the Google ADK (Agent Development Kit) within the τ-bench framework.

## 1. Clone the Repository

First, clone the specific version of the τ-bench repository that is compatible with these modifications.

```bash
git clone https://github.com/sierra-research/tau2-bench.git
cd tau2-bench
git checkout cc97b34
```

## 2. Setup and Installation

Follow the standard installation instructions to set up the environment for the project.


Ensure that you have venv:

```bash
sudo apt install python3.13-venv
```

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -e .
```

You can check tau data:

```bash
tau2 check-data
```

After the standard installation, you need to install the Google ADK package.

```bash
pip install google-adk
```

tenacity dependecy version may conflict with that of tau2 repo. Upgrading it back in favor of tau2.

```bash
pip install --upgrade tenacity
```

**IMPORTANT:** Gemini 3 Pro model makes sending thought signatures mandatory. Tau2 bench relies on litellm for user simulation and non-adk agent simulation. Until https://github.com/BerriAI/litellm/pull/16812 is merged to litellm repository, the PR needs to be applied as shown below:

```bash
git clone --filter=blob:none --quiet https://github.com/BerriAI/litellm.git /tmp/litellm-pr-16812
cd /tmp/litellm-pr-16812
git checkout -q pull/16812/head
git fetch origin pull/16812/head:pr-16812
git checkout pr-16812
pip install .
cd -
```

## 3. Add env params

Create `.env` file at root with the following content.

```bash
GOOGLE_GENAI_USE_VERTEXAI=true
GOOGLE_CLOUD_PROJECT=your_project_id
GOOGLE_CLOUD_LOCATION=global
VERTEXAI_LOCATION=global
```

## 4. Copy Modified Files

You will need to copy two files into your local `tau2-bench` directory. These files contain the implementation of the ADK agent and its unit tests.

Assuming you have the following files available:

*   `tau2_agent/adk_agent.py`
*   `tests/test_adk_agent.py`

Copy them to the correct locations within the project:

```bash
cp ../tau2_agent/adk_agent.py src/tau2/agent/adk_agent.py
cp ../tests/test_adk_agent.py tests/test_adk_agent.py
```

## 4.1. Registering the ADK Agent

To enable the `adk_agent` within the `tau2-bench` framework, you need to manually modify the `src/tau2/registry.py` file in your `tau2-bench` repository.

1.  **Add the import statement** for `AdkAgent` at the top of `src/tau2/registry.py`:
    ```python
    from tau2.agent.adk_agent import AdkAgent
    ```
2.  **Register the agent** within the `try` block where other default components are registered (look for `registry = Registry()`):
    ```python
    try:
        registry = Registry()
        logger.debug("Registering default components...")
        # ... existing registrations ...
        registry.register_agent(AdkAgent, "adk_agent") # Add this line
        # ... more existing registrations ...
    ```
    This allows the `adk_agent` to be selected via command-line arguments (e.g., `--agent adk_agent`).

## 5. Running the ADK Agent

Once the files are in place and dependencies are installed, you can run a simulation using the `adk_agent`.

### Changing baseline ADK agent

You can implement an improved agent and replace `AdkLlmAgent` implementation `_create_agent` returns as shown below. Return type is BaseAgent which accommodates workflow agents as well.

```py
def _create_agent(name: str, model: Union[str, BaseLlm], instruction: str, tools: List[Tool]) -> BaseAgent:
    adk_tools = [
        AdkTool(
            types.FunctionDeclaration(
                name=tool.openai_schema['function']['name'],
                description=tool.openai_schema['function'].get('description', ''),
                parameters_json_schema=tool.openai_schema['function']['parameters'],
            )
        )
        for tool in tools
    ]
    return AdkLlmAgent(
        model=model,
        name=name,
        instruction=instruction,
        tools=adk_tools,
        planner=built_in_planner.BuiltInPlanner(
            thinking_config=types.ThinkingConfig(include_thoughts=True),
        ),
    )
```

### Limited run

Here is an example command to run the agent on an airline domain task:

```bash
tau2 run --domain airline --agent adk_agent --agent-llm vertex_ai/gemini-3-pro-preview --user-llm vertex_ai/gemini-3-pro-preview --num-trials 1 --num-tasks 1 --user-llm-args '{"temperature": 1, "reasoning_effort": "high"}' --agent-llm-args '{"temperature": 1, "reasoning_effort": "high"}'
```

Optionally, you can run specific example by using `--task-ids` instead of `--num-tasks`.

**temperature:** When adk_agent is used defaults to 1. The commands in this document sets them explicitly using llm_args for both user and agent models.

**reasoning_level** Only applies to Gemini 3 Pro model. It defaults to high for adk_agent while using this model. Otherwise, it will default to dynamic thinking. Again this document demonsrates setting it explicitly using llm_args.

**NOTE**: It is normal that you will be getting `This model isn't mapped yet` error logs. This is coming from litellm cost calculation workflow used by `--user-llm`. You can suppress is temporarily by swapping `--user-llm vertex_ai/gemini-3-pro-preview` with `--user-llm vertex_ai/gemini-2.5-pro`.

### Viewing trajectories

You can use the following command to view trajectories after following the default options:

```bash
tau2 view
```

### Full run

Full run requires dropping the arg `--task-ids`.

```bash
# Example: Run complete evaluation for all domains
tau2 run \
  --domain retail \
  --agent adk_agent \
  --agent-llm vertex_ai/gemini-3-pro-preview \
  --user-llm vertex_ai/gemini-3-pro-preview \
  --num-trials 4 \
  --save-to gemini_3_pro_retail \
  --user-llm-args '{"temperature": 1, "reasoning_effort": "high"}' \
  --agent-llm-args '{"temperature": 1, "reasoning_effort": "high"}'


tau2 run \
  --domain airline \
  --agent adk_agent \
  --agent-llm vertex_ai/gemini-3-pro-preview \
  --user-llm vertex_ai/gemini-3-pro-preview \
  --num-trials 4 \
  --save-to gemini_3_pro_airline \
  --user-llm-args '{"temperature": 1, "reasoning_effort": "high"}' \
  --agent-llm-args '{"temperature": 1, "reasoning_effort": "high"}'


tau2 run \
  --domain telecom \
  --agent adk_agent \
  --agent-llm vertex_ai/gemini-3-pro-preview \
  --user-llm vertex_ai/gemini-3-pro-preview \
  --num-trials 4 \
  --save-to gemini_3_pro_telecom \
  --user-llm-args '{"temperature": 1, "reasoning_effort": "high"}' \
  --agent-llm-args '{"temperature": 1, "reasoning_effort": "high"}'
```

### Prepare Submission Package

```bash
tau2 submit prepare data/tau2/simulations/gemini_3_pro_*.json --output ./gemini_3_pro_submission
```

This command will:

- Verify all trajectory files are valid
- Check that submission requirements are met
- Compute performance metrics (Pass^k rates)
- Prompt for required metadata (model name, organization, contact email) -> you can pass dummy values here as we are not submitting yet.
- Create a structured submission directory with:
  - `submission.json`: Metadata and metrics
  - `trajectories/`: Your trajectory files

## 6. Testing the Agent

To verify that the agent is set up correctly, you can run its unit tests using `pytest`.

```bash
pytest tests/test_adk_agent.py
```

To see coverage (optional):

```bash
pip install pytest-cov
```

```bash
pytest --cov=tau2.agent.adk_agent --cov-report=html tests/test_adk_agent.py
````
