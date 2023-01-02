# execute the following code once to get a worker_id
from lightly.api import ApiWorkflowClient

client = ApiWorkflowClient()  # replace this with your token
worker_id = client.register_compute_worker()
print(f"Worker ID: {worker_id}")
