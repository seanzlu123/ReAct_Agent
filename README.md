-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
🦾 Minimal ReAct-style LLM Agent for AppWorld

This repository implements two versions of a ReAct-style autonomous LLM agent for AppWorld.
Choose the Minimal version for simplicity/rapid prototyping, or the Modular version for advanced, robust research workflows.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
📖 Table of Contents
--Overview
--Features Comparison
--Minimal Version (SimpleReAct)
----Getting Started
----How It Works
----Example Output
--Modular Version (ModularReAct)
----Getting Started
----How It Works
----Example Output
--Customization
--Experimental Analysis
--Acknowledgments 
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
📝 Overview
--MinimalReAct: A single-file, quick-start implementation. Best for those new to ReAct agents or looking for quick, straightforward prototypes.
--ModularReAct: A research-ready, robust, and extensible package. Modular design supports best-of-N logic, experience replay, error handling, logging, parallel runs, and more.
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
🔍 Features Comparison

Feature	                                       MinimalReAct	          ReActX
Modular Design	                                   ❌ (monolithic)   	✅ (fully modular)
Robust Error Handling	                           ❌	                ✅ (retries, logging)
Multi-run (“Best of N”) Logic	                   ❌	                ✅
Experience Memory/Few-shot	                       ❌	                ✅
Prompt Engineering (Manual)	                       ❌                   ✅	Contextual, dynamic
Experiment Configuration	                       ❌ (globals/inline)  ✅ (config object)
Parallelization	                                   ✅ ProcessPool       ✅ ThreadPool
Progress/Resource Monitoring	                   ❌Minimal	        ✅ (rich, psutil)
Difficulty Filtering                               ✅	                ❌ 
Result Summarization/History	                   ✅ (per task)       	✅ (best-of-N, all runs)

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
🚦 Minimal Version (SimpleReAct)

File: minimal_react_agent.py

🛠️ Getting Started
Python 3.9+
openai, jinja2, appworld (see AppWorld docs)

Installation
--pip install openai loguru rich psutil jinjia2 
# Follow AppWorld installation guide for setup

API Setup
Edit the API section in your code:

  python
  client = OpenAI(
      api_key="sk-your-api-key",
      base_url="https://your-openai-compatible-endpoint/v1"
  )

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Warning: Never commit your real API key to public repositories.

Ensure your AppWorld data/tasks are configured (see their docs).

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
📚 How It Works:  
1. Prompt Construction:
    Each task gets a Jinja2-rendered, in-context prompt with tool documentation and guidelines.
2. LLM Agent Loop:
   The agent uses conversation history to prompt the LLM, receives code, executes via AppWorld, and logs all turns.
3. Difficulty-aware Task Filtering 🆕:
   At the start of each run, you’re prompted for the target difficulty (1-easy, 2-medium, 3-hard).
   Each task's true difficulty (from its metadata) is loaded before evaluation. Only matching tasks are run, keeping experiments targeted and compute efficient.
4. Parallel Task Evaluation:
   Run multiple tasks at once via ProcessPoolExecutor for a real research workflow. 
5. Result Logging & Summarization:
   Reasoning steps, code, and results are saved per task (for tasks that pass filters). Summaries are printed by difficulty

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
📁 Code Structure: 
text
minimal_react_agent.py          # Main experiment script
│
├─ ReactAgent                   # Core LLM agent: prompt, message tracking, dialog
├─ sample_prompt                # Jinja2 template for ReAct context and tool docs
├─ evaluate_task                # Task loop: code gen, execution, eval, saving history/results
├─ get_task_difficulty 🆕       # Loads per-task metadata for pre-run filtering
├─ get_difficulty 🆕            # Prompts the user once for difficulty to run
├─ main (pool logic)            # Loads tasks, runs parallel evals, aggregates/filter results
│
└─ experiments/                 # All outputs, logs, and JSON histories by run


-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
🖼️ Example Output: 
text
Length of difficulty one: 3, Average Score: 0.5
Length of difficulty two: 9, Average Score: 0.34
Length of difficulty three: 3, Average Score: 0.5
More examples and detailed logs are printed during runs and saved per-task.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
🚀 Modular Version (ModularReAct)
See: main.py and the /llm_client, /World_client, /EM_client, /Summarizer, /tasks folders

🛠️ Getting Started
Python 3.9+
openai, loguru, rich, psutil

Installation
pip install openai loguru rich psutil
# install AppWorld, and ensure proper config in submodules

API Setup
Edit API key/base URL in llm_client/llm_client.py or inject via config/env vars.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
📚 How It Works:
1. Prompt: 
   Loads initial context for each task dynamically; can inject prior experience for few-shot.
2. Agent: 
   Modular ReactAgent uses a pluggable LLMClient—with retry, error handling, and logging.
3. Multiple Runs / Best-of-N:
   Runs each task up to N times (“best_at”), keeps the trajectory with highest score.
4. Experience Memory: 
   If enabled, summarizes/builds experience and injects into prompt for advanced training loops.
5. Parallel Task Running:
   ThreadPoolExecutor/rich.progress for live status. Resource usage monitored (psutil).
6. Extensible:
   All logic is in swappable modules (LLMClient, EnvClient, EMClient, etc).

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
🖼️ Example Output:
task_id: 12345 
difficulty: 2 
Score: 1 out of [1, 1, 0, 1] 

Difficulty 1: 22 tasks, Average Score: 0.80
Difficulty 2: 40 tasks, Average Score: 0.45
Difficulty 3: 28 tasks, Average Score: 0.13
Overall Average: 0.47
Best of: 4

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
📁 Structure
Simple_ReAct 
├── Config/
├── EM_client/
├── experiments/
├── llm_client/
├── Summarizer/
├── tasks/
└── World_client/

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
🧩 Customization: 
--Model/LLM Endpoint: Change model or API endpoint in LLMClient instantiation.
--Prompt/context logic: MinimalReAct uses Jinja2 templates; ModularReAct can use summarized experiences/few-shot.
--Experiment scale: Set sample_size, max_interactions, best_at (Modular), max_workers.
--Experience Replay: Modular version only—turn on run_with_experience and create_exp.
--Logging/Filtering: Both support difficulty filtering, ModularReAct supports full per-task and per-run tracing.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
📊 Experimental Analysis
--Both agents report average scores by difficulty/group.
--ModularReAct adds “best-of”, experience memory, richer error handling/logging, and real-time progress bar with memory stats.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
🏷️ Acknowledgments
Inspired by the LLM tool-use (ReAct) paradigm, OpenAI, the AppWorld research platform, and the autonomous agent research community.
