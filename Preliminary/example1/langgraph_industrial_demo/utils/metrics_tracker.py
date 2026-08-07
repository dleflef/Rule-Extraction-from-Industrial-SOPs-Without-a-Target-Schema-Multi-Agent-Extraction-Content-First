import time
import json
import tiktoken
from functools import wraps

class MetricsTracker:
    # Tracks token usage, execution time, and data size for benchmark reporting."""
    def __init__(self):
        self.metrics = []
        # Using cl100k_base which is a close approximation for Llama/OpenAI tokenization
        try:
            self.encoder = tiktoken.get_encoding("cl100k_base")
        except:
            self.encoder = None

    def count_tokens(self, text):
        # Convert text into tokens using tiktoken if available, otherwise fallback to a rough estimation
        if not self.encoder or not isinstance(text, str):
            # Fallback estimation (roughly 4 chars per token) if tiktoken fails
            return len(str(text)) // 4
        return len(self.encoder.encode(str(text)))

    def log_metric(self, agent_name, exec_time, prompt_tokens, completion_tokens, input_size, output_size):
        self.metrics.append({
            "agent_name": agent_name,
            "execution_time_seconds": round(exec_time, 4),
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "input_size_bytes": input_size,
            "output_size_bytes": output_size
        })

    def save_metrics(self, filepath="benchmark_metrics.json"):
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=4)

tracker = MetricsTracker()

def track_performance(agent_name):
    """Decorator to track real latency and local token estimation."""
    def decorator(func):
        @wraps(func)
        def wrapper(state, *args, **kwargs):
            start_time = time.time()
            # How much input data is being processed (in bytes)
            input_text = str(state)
            input_size = len(input_text)
            
            # Remove LangGraph internal arguments
            kwargs.pop('config', None)
            kwargs.pop('store', None)
            kwargs.pop('writer', None)
            
            # Execute the function
            result = func(state, *args, **kwargs)
                
            exec_time = time.time() - start_time
            output_text = str(result)
            output_size = len(output_text)
            
            # ESTIMATE REAL TOKENS LOCALLY
            prompt_tokens = tracker.count_tokens(input_text)
            completion_tokens = tracker.count_tokens(output_text)
            
            tracker.log_metric(
                agent_name, 
                exec_time, 
                prompt_tokens, 
                completion_tokens, 
                input_size, 
                output_size
            )
            return result
        return wrapper
    return decorator
