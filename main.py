from autogen import UserProxyAgent, config_list_from_json
from autogen.agentchat.contrib.teachable_agent import TeachableAgent

import os
import sys

try:
    from termcolor import colored
except ImportError:

    def colored(x, *args, **kwargs):
        return x

config_list = config_list_from_json(
    env_or_file="OAI_CONFIG_LIST", 
    filter_dict={
        "model": [
            "gpt-3.5-turbo-1106",
        ]
    }
)

cache_seed = None  # Use an int to seed the response cache. Use None to disable caching.

llm_config={
    "config_list": config_list, 
    "timeout": 120, 
    "cache_seed": cache_seed
}

verbosity = 0  # 0 for basic info, 1 to add memory operations, 2 for analyzer messages, 3 for memo lists.
recall_threshold = 1.5  # Higher numbers allow more (but less relevant) memos to be recalled.

teachable_agent = TeachableAgent(
    name="teachableagent",
    llm_config=llm_config,
    teach_config={
        "verbosity": verbosity,
        "recall_threshold": recall_threshold,
        "path_to_db_dir": "./tmp/interactive/teachable_agent_db",
        "reset_db": False,
    },
)

# Create the agents.
print(colored("\nLoading previous memory (if any) from disk.", "light_cyan"))
user = UserProxyAgent("user", human_input_mode="ALWAYS")

# Start the chat.
teachable_agent.initiate_chat(user, message="Greetings, I'm a teachable user assistant! What's on your mind today?")

# Let the teachable agent remember things that should be learned from this chat.
teachable_agent.learn_from_user_feedback()

# Wrap up.
teachable_agent.close_db()