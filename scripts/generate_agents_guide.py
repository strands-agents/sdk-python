#!/usr/bin/env python3
"""
Generate AGENTS.md documentation using a Strands Agent.

This script creates a Strands Agent with file reading and writing capabilities
and uses it to generate comprehensive documentation for AI agents.
"""

from strands import Agent
from strands_tools import file_read, editor


def main():
    """Generate the AGENTS.md documentation."""
    # Create agent with file system tools
    doc_agent = Agent(tools=[file_read, editor])
    
    # Load the generation prompt
    with open('scripts/generate_agents_guide.md', 'r') as f:
        prompt = f.read()
    
    print("Generating AGENTS.md documentation...")
    
    # Generate the documentation
    result = doc_agent(prompt)
    
    # Write result to AGENTS.md file
    with open('AGENTS.md', 'w') as f:
        f.write(str(result))
    
    print("Documentation generation completed! Written to AGENTS.md")


if __name__ == "__main__":
    main()
