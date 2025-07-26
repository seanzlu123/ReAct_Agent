import requests
from typing import Dict, List, Optional, Any
import json


class EnvClient:
    def __init__(self, base_url: str = "http://localhost:9000"):
        self.base_url = base_url.rstrip("/")

    def _make_request(
        self,
        endpoint: str,
        env_type: str = "default",
        task_id: str = None,
        instance_id: str = None,
        messages: Dict[str, Any] = None,
        params: Dict[str, Any] = None,
    ) -> Dict:
        """Unified request handling method"""
        url = f"{self.base_url}/{endpoint}"
        data = {
            "env_type": env_type,
            "task_id": task_id,
            "instance_id": instance_id,
            "messages": messages or {},
            "params": params or {},
        }
        response = requests.post(url, json=data)
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            # Directly expose the detail returned by the backend FastAPI
            try:
                detail = response.json().get("detail", "")
            except Exception:
                detail = response.text
            raise Exception(f"{e} – {detail}")
        return response.json()

    def     get_task_ids(
        self, env_type: str, split: str = "dev", params: dict | None = None
    ) -> List[str]:
        """Get task ID list"""
        payload = {"env_type": env_type}
        if params:
            payload["params"] = params
        response = self._make_request(
            endpoint="get_task_ids", env_type=env_type, params={"split": split}
        )
        return response["data"]

    def get_tools_info(
        self, instance_id: str, messages: Dict = {}, params: Dict = {}
    ) -> float:
        """Get environment information"""
        response = self._make_request(
            endpoint="get_info",
            instance_id=instance_id,
            messages=messages,
            params=params,
        )
        return response["data"]

    def create_instance(
        self, env_type: str, task_id: str, instance_id: str = None, params: Dict = None
    ) -> Dict[str, str]:
        """Create environment instance"""
        response = self._make_request(
            endpoint="create",
            env_type=env_type,
            task_id=task_id,
            instance_id=instance_id,
            params=params,
        )
        return response["data"]

    def step(self, instance_id: str, action: Dict = {}, params: Dict = {}) -> str:
        """Execute environment step"""
        response = self._make_request(
            endpoint="step", instance_id=instance_id, messages=action, params=params
        )
        return response["data"]

    def evaluate(
        self, instance_id: str, messages: Dict = {}, params: Dict = {}
    ) -> float:
        """Evaluate environment instance"""
        response = self._make_request(
            endpoint="evaluate",
            instance_id=instance_id,
            messages=messages,
            params=params,
        )
        return response["data"]

    def release_instance(self, instance_id: str) -> bool:
        """Release environment instance"""
        response = self._make_request(endpoint="release", instance_id=instance_id)
        return response["success"]


# Usage example
def main():
    client = EnvClient()

    env_type = "appworld"
    # Get task list
    task_ids = client.get_task_ids(env_type)
    print(f"Available tasks: {task_ids}")

    # Create instance
    task_id = task_ids[0]
    init_response = client.create_instance(env_type, task_id)
    print("init state", init_response)
    instance_id = init_response["info"]["instance_id"]
    query = init_response["state"]
    print(f"Created instance {instance_id} with query: {query}")

    # Execute action
    action = {"role": "assistant", "content": "print('hello appworld!!')"}
    result = client.step(instance_id, action)
    print(f"Step result: {result}")

    # Evaluate
    score = client.evaluate(instance_id)
    print(f"Evaluation score: {score}")

    # Release instance
    success = client.release_instance(instance_id)
    print(f"Instance released: {success}")


if __name__ == "__main__":
    main()
