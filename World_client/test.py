# sandbox_envclient.py
from World_client.env_client import EnvClient
import json

def main():
    # ---- 1. Initialize EnvClient ----
    app_client = EnvClient(base_url="http://localhost:8003")

    # ---- 2. Get task ids ----
    env_type = "appworld"
    dataset_split = "train"

    task_ids = app_client.get_task_ids(env_type, split=dataset_split)
    print(f"Total tasks: {len(task_ids)}")
    if not task_ids:
        print("No tasks found! Check your backend server.")
        return

    # ---- 3. Create an instance ----
    task_id = task_ids[0]
    init_response = app_client.create_instance(env_type=env_type, task_id=task_id)
    print("\n--- Init response ---")
    print(json.dumps(init_response, indent=2))

    instance_id = init_response["info"]["instance_id"]
    instruction = init_response["state"]
    print(f"\nInstance ID: {instance_id}")
    print(f"Task Instruction:\n{instruction}")


    # ---- 4. (Optional) Echo tools info/info endpoint ----
    try:
        info = app_client.get_tools_info(instance_id)
        print(f"\nTools info:\n{json.dumps(info, indent=2)}")
    except Exception as e:
        print("No tools info endpoint or failed:", e)

    # ---- 5. Run a single step (send sample code/action) ----
    action = {
        "role": "assistant",
        "content": "print('Hello from EnvClient sandbox!')"
    }
    step_result = app_client.step(instance_id, action)
    print("\n--- Step Result ---")
    print(json.dumps(step_result, indent=2))

    # ---- 6. Evaluate the instance ----
    eval_result = app_client.evaluate(instance_id)
    print("\n--- Evaluation ---")
    print(json.dumps(eval_result, indent=2))

    # ---- 7. Release the instance ----
    released = app_client.release_instance(instance_id)
    print(f"\nInstance released: {released}")

if __name__ == "__main__":
    main()
