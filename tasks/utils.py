import os
import re
import json
from loguru import logger

def extract_task(content):
    """
    Extracts the task description from an input string using a regex.
    Logs a warning if no task instruction is found.
    """
    matches = re.findall(r'Task:\s*(.+)', content)
    if matches:
        return matches[-1].strip()
    else:
        logger.warning("No task instruction found")

def insert_context_before_task(init_content, context):
    #This regex replaces the LAST occurrence of a line starting with Task:

    pattern = r'(.*?)(\nTask: .*)$'   # non-greedy up to last "\nTask: ..."
    match = re.search(pattern, init_content, flags=re.DOTALL)
    if match:
        before = match.group(1).rstrip()
        task_line = match.group(2).rstrip()
        #Insert advice between 'before' and 'task_line'
        return f"{before}\n\n {context.strip()}\n\n {task_line}"
    else:
        logger.info(f"No task instruction found")
        return f"{init_content.strip()}\n\n {context.strip()}\n"

def get_task_difficulty(task_id):
    """
    Retrieves the 'difficulty' value for a task_id by reading its metadata file.
    Logs errors for missing files or JSON read failures.
    """
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Project root directory
    metadata_path = os.path.join(BASE_DIR, 'Agent', 'data', 'tasks', task_id, 'ground_truth', 'metadata.json')
    if not os.path.exists(metadata_path):
        logger.error(f"Task {task_id} does not exist at path {metadata_path}")
        return None
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        return metadata.get("difficulty", None)
    except Exception as e:
        logger.exception(f"Failed to read difficulty for task {task_id}: {e}")
        return None