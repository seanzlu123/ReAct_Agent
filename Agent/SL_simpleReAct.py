# Import required libraries and modules
import re
import os
import json
import statistics
from openai import OpenAI
from jinja2 import Template
from dotenv import load_dotenv
from run_appworld import AppWorld
from collections import defaultdict
from run_appworld.task import Task, load_task_ids
from concurrent.futures import ProcessPoolExecutor, as_completed


# This is the base system/user/assistant prompt template fed to LLM for each task,
# including grounding instructions, tool use examples, and detailed guidelines.
sample_prompt = """
USER:
I am your supervisor and you are a super intelligent AI Assistant whose job is to achieve my day-to-day tasks completely autonomously.

To do this, you will need to interact with app/s (e.g., spotify, venmo, etc) using their associated APIs on my behalf. For this you will undertake a *multi-step conversation* using a python REPL environment. That is, you will write the python code and the environment will execute it and show you the result, based on which, you will write python code for the next step and so on, until you've achieved the goal. This environment will let you interact with app/s using their associated APIs on my behalf.

Here are three key APIs that you need to know to get more information

# To get a list of apps that are available to you.
print(apis.api_docs.show_app_descriptions())

# To get the list of apis under any app listed above, e.g. supervisor
print(apis.api_docs.show_api_descriptions(app_name='supervisor'))

# To get the specification of a particular api, e.g. supervisor app's show_account_passwords
print(apis.api_docs.show_api_doc(app_name='supervisor', api_name='show_account_passwords'))
Each code execution will produce an output that you can use in subsequent calls. Using these APIs, you can now generate code, that the environment will execute, to solve the task.

For example, consider the task:

My name is: {{ supervisor.first_name }} {{ supervisor.last_name }}. My personal email is {{ supervisor.email }} and phone number is {{ supervisor.phone_number }}.

Task:

What is the password for my Spotify account?

ASSISTANT:
# Okay. Lets first find which apps are available to get the password by looking at the app descriptions.
print(apis.api_docs.show_app_descriptions())

USER:
[
  {
    "name": "api_docs",
    "description": "An app to search and explore API documentation."
  },
  {
    "name": "supervisor",
    "description": "An app to access supervisor's personal information, account credentials, addresses, payment cards, and manage the assigned task."
  },
  ...
  {
    "name": "spotify",
    "description": "A music streaming app to stream songs and manage song, album and playlist libraries."
  },
  {
    "name": "venmo",
    "description": "A social payment app to send, receive and request money to and from others."
  },
  ...
]
ASSISTANT:
# Looks like the supervisor app could help me with that. Lets see what apis are available under this app.
print(apis.api_docs.show_api_descriptions(app_name='supervisor'))


USER:
[
  ...
  "show_account_passwords : Show your supervisor's account passwords."
  ...
]


ASSISTANT:
# I can use `show_account_passwords` to get the passwords. Let me see its detailed specification to understand its arguments and output structure.
print(apis.api_docs.show_api_doc(app_name='supervisor', api_name='show_account_passwords'))

USER:
{
'app_name': 'supervisor',
  'api_name': 'show_account_passwords',
  'path': '/account_passwords',
  'method': 'GET',
  'description': "Show your supervisor's app account passwords.",
  'parameters': [],
  'response_schemas': {
    'success': [{'account_name': 'string', 'password': 'string'}],
    'failure': {'message': 'string'}
  }
}  
ASSISTANT:
# Okay, it requires no arguments. So I can just call it directly.
print(apis.supervisor.show_account_passwords())

USER:
[
  {
    "account_name": "spotify",
    "password": "dummy_spotify_pass"
  },
  {
    "account_name": "file_system",
    "password": "dummy_fs_pass"
  },
  ...
]
ASSISTANT:
# So the Spotify password is an entry in the `passwords` list with the account_name=spotify.
spotify_password = [account_password["account_name"] == "spotify" for account_password in passwords][0]["password"]
print(spotify_password)


USER:
dummy_spotify_pass

ASSISTANT:
# When the task is completed, I need to call apis.supervisor.complete_task(). If there is an answer, I need to pass it as an argument `answer`. I will pass the spotify_password as an answer.
apis.supervisor.complete_task(answer=spotify_password)


USER:
Marked the active task complete.


----------------------------------------------

USER:
**Key instructions and disclaimers**:
1. The email addresses, access tokens and variables (e.g. spotify_password) in the example above were only for demonstration. Obtain the correct information by calling relevant APIs yourself.
2. Only generate valid code blocks, i.e., do not put them in ```...``` or add any extra formatting. Any thoughts should be put as code comments.
3. You can use the variables from the previous code blocks in the subsequent code blocks.
4. Write small chunks of code and only one chunk of code in every step. Make sure everything is working correctly before making any irreversible change.
5. The provided Python environment has access to its standard library. But modules and functions that have a risk of affecting the underlying OS, file system or process are disabled. You will get an error if do call them.
6. Any reference to a file system in the task instructions means the file system *app*, operable via given APIs, and not the actual file system the code is running on. So do not write code making calls to os-level modules and functions.
7. To interact with apps, only use the provided APIs, and not the corresponding Python packages. E.g., do NOT use `spotipy` for Spotify. Remember, the environment only has the standard library.
8. The provided API documentation has both the input arguments and the output JSON schemas. All calls to APIs and parsing its outputs must be as per this documentation.
9. For APIs that return results in "pages", make sure to consider all pages.
10. To obtain current date or time, use Python functions like `datetime.now()` or obtain it from the phone app. Do not rely on your existing knowledge of what the current date or time is.
11. For all temporal requests, use proper time boundaries, e.g., if I ask for something that happened yesterday, make sure to consider the time between 00:00:00 and 23:59:59. All requests are concerning a single, default (no) time zone.
12. Any reference to my friends, family or any other person or relation refers to the people in my phone's contacts list.
13. All my personal information, and information about my app account credentials, physical addresses and owned payment cards are stored in the "supervisor" app. You can access them via the APIs provided by the supervisor app.
14. Once you have completed the task, call `apis.supervisor.complete_task()`. If the task asks for some information, return it as the answer argument, i.e. call `apis.supervisor.complete_task(answer=<answer>)`. For tasks that do not require an answer, just skip the answer argument or pass it as None.
15. The answers, when given, should be just entity or number, not full sentences, e.g., `answer=10` for "How many songs are in the Spotify queue?". When an answer is a number, it should be in numbers, not in words, e.g., "10" and not "ten".
16. You can also pass `status="fail"` in the complete_task API if you are sure you cannot solve it and want to exit.
17. You must make all decisions completely autonomously and not ask for any clarifications or confirmations from me or anyone else.

USER:
Using these APIs, now generate code to solve the actual task:

My name is: {{ supervisor.first_name }} {{ supervisor.last_name }}. My personal email is {{ supervisor.email }} and phone number is {{ supervisor.phone_number }}.

Task:

{{ instruction }}
"""

load_dotenv("../.env")
api_key = os.getenv("OPENAI_API_KEY")
experiment_name = "test_5b"

# ----------- OpenAI API Client Init -----------
client = OpenAI()

#NEW: Prompt user for which difficulty level to use
def get_difficulty():
    desired_difficulty = int(input("Enter desired difficulty [1-3]: "))
    return desired_difficulty

#NEW: Retrieve the difficulty for a given task before running it
def get_task_difficulty(task_id):
    with open(f"data/tasks/{task_id}/ground_truth/metadata.json") as f:
        return json.load(f)["difficulty"]

eval_before = True  #Evalute tasks before Interacting
filter_by_difficulty = True #Filter tasks by difficulty


def call_llm(messages: list[dict]) -> str:
    """
    Call an LLM API to get the next assistant message for a given history.

    Args:
        messages: List of dicts with 'role' and 'content' representing the conversation so far.
    Returns:
        The LLM's response string.
    """
    response = client.chat.completions.create(
        model="qwen-max", messages=messages, temperature=0.0, max_tokens=400, seed=123
    )
    text_response = ""
    if response.choices:
        text_response = response.choices[0].message.content
    return text_response


class ReactAgent:
    """
    Holds all agent logic: prompt build and message tracking per task.
    """

    def __init__(self, task: Task):
        self.task = task
        self.history: list[dict] = self.prompt_messages()  # Seed message history with the system/user context

    def prompt_messages(self) -> list[dict]:
        """
        Create and split the full task-specific LLM prompt into OpenAI-style message blocks.
        Returns:
            A list of {'role': str, 'content': str} dicts, compatible with OpenAI API format.
        """
        # Prepare values and render prompt
        dictionary = {"supervisor": self.task.supervisor, "instruction": self.task.instruction}
        prompt = Template(sample_prompt.lstrip()).render(dictionary)

        # Split the prompt into distinct interaction turns
        messages: list[dict] = []
        last_start = 0
        for match in re.finditer("(USER|ASSISTANT|SYSTEM):\n", prompt):
            last_end = match.span()[0]
            if len(messages) == 0:
                if last_end != 0:
                    raise ValueError(f"Start of the prompt has no assigned role: {prompt[:last_end]}")
            else:
                messages[-1]["content"] = prompt[last_start:last_end]
            mesg_type = match.group(1).lower()
            messages.append({"role": mesg_type, "content": None})
            last_start = match.span()[1]
        messages[-1]["content"] = prompt[last_start:]
        return messages

    def next_code_block(self, last_execution_output: str | None = None) -> str:
        """
        Ask the LLM for the next code block, using the current conversation history and last code's output.
        Args:
            last_execution_output: What happened after the last code (str or None)
        Returns:
            Code string to execute next.
        """
        if last_execution_output is not None:
            self.history.append({"role": "user", "content": last_execution_output})
        code = call_llm(self.history)
        self.history.append({"role": "assistant", "content": code})
        return code


def evaluate_task(task_id: str, experiment_name: str, max_interactions: int, count: int, sample_size: int, filter_on: bool, difficulty: int) -> dict:
    """
    Execute one full task experiment (prompt→REPL→score) and collect results for summary.
    Args:
        task_id: ID of the task to solve
        experiment_name: Name to pass to AppWorld
        max_interactions: Max REPL rounds allowed per task
    Returns:
        Dict with task_id, difficulty, and score
    """
    with AppWorld(
            task_id=task_id,
            experiment_name=experiment_name,
    ) as world:
        print("\n\n" + "*" * 20 + f" Task: {count} | {sample_size} " + "*" * 20)
        print(world.task.instruction)
        agent = ReactAgent(world.task)
        output: str | None = None

        #NEW: retrieve task difficulty beforehand
        task_difficulty = get_task_difficulty(task_id)

        #MODIFIED: Step through the agent/LLM-REPL loop if eval_before is False, Additional Filter option based on difficulty
        if not eval_before:
            if (not filter_on) or (task_difficulty == difficulty):
                for _ in range(max_interactions):
                    print(f"Interaction: {_}")
                    code = agent.next_code_block(output)
                    print("\n\n" + "%" * 20 + " CODE " + "%" * 20 + "\n" + code)
                    output = world.execute(code)
                    print("\n\n" + "=" * 20 + " OUTPUT " + "=" * 20 + "\n" + output)
                    if world.task_completed():
                        print("Task completed")
                        break

        evaluations = world.evaluate().to_dict()

        num_success = len(evaluations["passes"])
        num_failure = len(evaluations["failures"])
        score = num_success / evaluations['num_tests']
        result = {
            'task_id': task_id,
            'difficulty': evaluations['difficulty'],
            'score': score
        }

    # Print evaluation summary for this task
    print(f"Evaluation: {evaluations} \n "
          f"Success: {num_success} \n"
          f" Failure: {num_failure} \n "
          f"Difficulty: {evaluations['difficulty']} \n"
          f"Score: {score} \n"
          f"History: {agent.history}")

    history_dir = os.path.join("experiments", f"{experiment_name}")
    output_filename = os.path.join(history_dir, f"history_{experiment_name}_{task_id}.json")

    #NEW: Only write history file if passed filter, mirrors LLM loop logic
    if () or (evaluations['difficulty'] == difficulty):
        with open(output_filename, "w") as f:
            json.dump(agent.history, f, indent=2)
    return result


# ===== Parallel experiment (multi-process) main logic =====
if __name__ == "__main__":
    # ----- Experiment settings -----
    sample_size = 90
    dataset_name = "train"
    max_interactions = 50

    # ----- Load all task IDs to process -----
    task_ids = load_task_ids(dataset_name)

    all_results = []

    # ----- Run tasks in parallel using process pool (avoid SQLite threading issues) -----
    with ProcessPoolExecutor(max_workers=10) as executor:
        #NEW: User-desired filter, if filtering is enabled
        if filter_by_difficulty:
            difficulty = get_difficulty()
        # Submit all tasks
        futures = [
            executor.submit(evaluate_task, task_id, experiment_name + str(idx), max_interactions, idx+1, sample_size, filter_by_difficulty, difficulty)
            for idx,task_id in enumerate(task_ids[:sample_size])
        ]

        # MODIFIED: Results filtered ONLY if filtering is on and result's difficutly mismatches
        for future in as_completed(futures):
            result = future.result()
            if filter_by_difficulty and result['difficulty'] != difficulty:
                continue
            all_results.append(result)


    # ----- Aggregate results by difficulty -----
    difficulty_one = [r for r in all_results if r['difficulty'] == 1]
    difficulty_two = [r for r in all_results if r['difficulty'] == 2]
    difficulty_three = [r for r in all_results if r['difficulty'] == 3]


    def avg(lst):
        """Compute average score for a list of result dicts."""
        return sum(r['score'] for r in lst) / len(lst) if lst else 0.0


    difficulty_scores = defaultdict(list)
    for res in all_results:
        if 'difficulty' in res and 'score' in res:
            difficulty_scores[res['difficulty']].append(res['score'])

    for diff, scores in sorted(difficulty_scores.items()):
        avg = statistics.mean(scores) if scores else 0
        print(f"Difficulty {diff}: {len(scores)} tasks, Average Score: {avg}")

    print(f"Overall Average: {statistics.mean([r['score'] for r in all_results])}")