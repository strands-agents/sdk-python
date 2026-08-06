#!/usr/bin/env python3
"""
# Knowledge Base Agent

This example demonstrates a Strands agent that can intelligently determine
whether to store information to a knowledge base or retrieve information from it
based on the user's query.

## Key Features
- Clear determination of store vs. retrieve actions
- Simple, user-friendly interactions
- Clean, readable output

## How to Run
1. Navigate to the example directory
2. Run: python knowledge_base_agent.py
3. Enter queries or information at the prompt

## Example Queries
- "Remember that my birthday is on July 25"
- "What day is my birthday?"
- "The capital of France is Paris"
"""

import asyncio
import os
from strands import Agent
from strands.vended_memory_stores import BedrockKnowledgeBaseStore
from strands_tools import use_llm

# Set knowledge base ID (can be overridden by environment variable)
DEFAULT_KB_ID = "demokb123"
KB_ID = os.environ.get("STRANDS_KNOWLEDGE_BASE_ID")

# Data source to ingest into. Only CUSTOM and S3 data sources accept writes.
DATA_SOURCE_ID = os.environ.get("STRANDS_DATA_SOURCE_ID", "demods123")

if not KB_ID:
    print("\n⚠️  Warning: STRANDS_KNOWLEDGE_BASE_ID environment variable is not set!")
    print("To use a real knowledge base, please set the STRANDS_KNOWLEDGE_BASE_ID environment variable.")
    print("For example: export STRANDS_KNOWLEDGE_BASE_ID=your_kb_id")
    print(f"Using DEFAULT_KB_ID '{DEFAULT_KB_ID}' for demonstration purposes only.")
    KB_ID = DEFAULT_KB_ID
else:
    print(f"Using Knowledge Base ID: {KB_ID}")

# The store is called directly rather than registered as a tool, so the workflow
# below decides when to read and when to write.
store = BedrockKnowledgeBaseStore(
    name="knowledge_base",
    description="Facts the user has asked to remember.",
    writable=True,
    config={
        "knowledge_base_id": KB_ID,
        "data_source_type": "CUSTOM",
        "data_source_id": DATA_SOURCE_ID,
    },
)

# System prompt to determine action
ACTION_SYSTEM_PROMPT = """
You are a knowledge base assistant focusing ONLY on classifying user queries.
Your task is to determine whether a user query requires STORING information to a knowledge base
or RETRIEVING information from a knowledge base.

Reply with EXACTLY ONE WORD - either "store" or "retrieve".
DO NOT include any explanations or other text.

Examples:
- "Remember that my birthday is July 4" -> "store"
- "What's my birthday?" -> "retrieve"
- "The capital of France is Paris" -> "store"
- "What is the capital of France?" -> "retrieve"
- "My name is John" -> "store"
- "Who am I?" -> "retrieve"
- "I live in Seattle" -> "store"
- "Where do I live?" -> "retrieve"

Only respond with "store" or "retrieve" - no explanation, prefix, or any other text.
"""

# System prompt for generating answers from retrieved information
ANSWER_SYSTEM_PROMPT = """
You are a helpful knowledge assistant that provides clear, concise answers
based on information retrieved from a knowledge base.

The information from the knowledge base contains the stored text of each match.
Focus on the actual content and ignore any metadata.

Your responses should:
1. Be direct and to the point
2. Not mention the source of information (like document IDs or scores)
3. Not include any metadata or technical details
4. Be conversational but brief
5. Acknowledge when information is conflicting or missing
6. Begin the response with \n

When analyzing the knowledge base results:
- Look for patterns across multiple results
- Ignore any JSON formatting or technical elements in the content

Example response for conflicting information:
"Based on my records, I have both July 4 and August 8 listed as your birthday. Could you clarify which date is correct?"

Example response for clear information:
"Your birthday is on July 4."

Example response for missing information:
"I don't have any information about your birthday stored."
"""

def determine_action(agent, query):
    """Determine if the query is a store or retrieve action."""
    result = agent.tool.use_llm(
        prompt=f"Query: {query}",
        system_prompt=ACTION_SYSTEM_PROMPT
    )

    # Clean and extract the action
    action_text = str(result).lower().strip()

    # Default to retrieve if response isn't clear
    if "store" in action_text:
        return "store"
    else:
        return "retrieve"

def run_kb_agent(query):
    """Process a user query with the knowledge base agent."""
    agent = Agent(tools=[use_llm])

    # Determine the action - store or retrieve
    action = determine_action(agent, query)

    if action == "store":
        # For store actions, store the full query. The store's methods are async,
        # so this sync workflow drives them with asyncio.run.
        asyncio.run(store.add(query))
        print("\nI've stored this information.")
    else:
        # For retrieve actions, query the knowledge base with appropriate parameters
        entries = asyncio.run(
            store.search(
                query,
                options={"max_search_results": 9},  # Limit number of results
            )
        )
        # Join the matched content to pass to the answering agent
        result_str = "\n".join(entry.content for entry in entries)

        # Generate a clear, conversational answer using the retrieved information
        answer = agent.tool.use_llm(
            prompt=f"User question: \"{query}\"\n\nInformation from knowledge base:\n{result_str}\n\nStart your answer with newline character and provide a helpful answer based on this information:",
            system_prompt=ANSWER_SYSTEM_PROMPT
        )

if __name__ == "__main__":
    # Print welcome message
    print("\n🧠 Knowledge Base Agent 🧠\n")
    print("This agent helps you store and retrieve information from your knowledge base.")
    print("Try commands like:")
    print("- \"Remember that my birthday is on July 25\"")
    print("- \"What day is my birthday?\"")
    print("\nType your request below or 'exit' to quit:")

    # Interactive loop
    while True:
        try:
            user_input = input("\n> ")
            if user_input.lower() in ["exit", "quit"]:
                print("\nGoodbye! 👋")
                break

            if not user_input.strip():
                continue

            # Process the input through the knowledge base agent
            print("Processing...")
            run_kb_agent(user_input)

        except KeyboardInterrupt:
            print("\n\nExecution interrupted. Exiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
