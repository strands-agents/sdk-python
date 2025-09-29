# Structured Output with Strands Agents

This guide demonstrates how to use structured output features with Strands Agents to get predictable, typed responses from language models using Pydantic models.

## Table of Contents

- [Overview](#overview)
- [Basic Usage](#basic-usage)
- [Advanced Features](#advanced-features)
- [Use Cases](#use-cases)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

Structured output allows you to define the exact format you want your AI agent to return using Pydantic models. Instead of parsing unstructured text responses, you get validated Python objects that match your specified schema.

### Key Benefits

- **Type Safety**: Get validated Python objects instead of raw text
- **Consistency**: Ensure responses always match your expected format  
- **Integration**: Easy integration with existing Python codebases
- **Validation**: Built-in data validation using Pydantic
- **Tool Compatibility**: Works seamlessly with existing Strands tools


## Basic Usage

### Simple Structured Output

```python
from strands import Agent
from pydantic import BaseModel

class UserProfile(BaseModel):
    """Basic user profile model."""
    name: str
    age: int
    occupation: str
    active: bool = True

# Create agent and use structured output
agent = Agent()
result = agent(
    "Create a profile for John Doe who is a 25 year old dentist",
    structured_output_type=UserProfile
)

# Access the structured result
profile = result.structured_output
print(profile.name)  # "John Doe"
print(profile.age)   # 25
```

### Regular Agent vs Structured Output

**Regular Agent (returns text)**:
```python
basic_agent = Agent()
basic_result = basic_agent("What can you do for me?")
# result.structured_output is None
# Response is in result.message
```

**Structured Output Agent**:
```python
agent = Agent()
result = agent(
    "Create a profile for Jake Johnson, age 28, software engineer",
    structured_output_type=UserProfile
)
# result.structured_output contains validated UserProfile object
```

## Advanced Features

### Complex Nested Models

```python
from typing import List, Optional
from pydantic import BaseModel, Field

class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str

class Contact(BaseModel):
    email: str
    phone: Optional[str] = None
    preferred_method: str = "email"

class Employee(BaseModel):
    """Complex nested employee model."""
    name: str
    employee_id: int
    department: str
    address: Address
    contact: Contact
    skills: List[str]
    hire_date: str
    salary_range: str

# Use the complex model
agent = Agent()
result = agent(
    "Create an employee profile for Sarah Smith, ID 12345, Engineering dept, "
    "living at 123 Main St, Seattle, WA 98101, email sarah@company.com, "
    "skills: Python, AWS, Docker, hired 2024-01-15, salary 80-120k",
    structured_output_type=Employee
)
```

### Validation and Constraints

```python
from pydantic import BaseModel, Field, field_validator

class ProductReview(BaseModel):
    """Product review with validation."""
    product_name: str
    rating: int = Field(ge=1, le=5, description="Rating from 1-5 stars")
    sentiment: str = Field(pattern="^(positive|negative|neutral)$")
    key_points: List[str]
    would_recommend: bool

class Name(BaseModel):
    first_name: str
    
    @field_validator("first_name")
    @classmethod
    def validate_first_name(cls, value: str) -> str:
        if not value.endswith('abc'):
            raise ValueError("You must append 'abc' to the end of my name")
        return value

# The agent will retry if validation fails
agent = Agent()
result = agent("What's Aaron's name?", structured_output_type=Name)
# Will automatically retry until validation passes
```

### Asynchronous Usage

```python
import asyncio
from strands import Agent

async def async_example():
    agent = Agent()
    
    result = await agent.invoke_async(
        """
        Analyze this product review:
        "This wireless mouse is fantastic! Great battery life, smooth tracking, 
        and the ergonomic design is perfect for long work sessions. The price 
        is reasonable too. I'd definitely buy it again and recommend it to others.
        Rating: 5 stars"
        """,
        structured_output_type=ProductReview
    )
    
    return result.structured_output

# Run async
review = asyncio.run(async_example())
```

### Streaming with Structured Output

```python
from strands import Agent

class WeatherForecast(BaseModel):
    location: str
    temperature: int
    condition: str
    humidity: int
    wind_speed: int
    forecast_date: str

agent = Agent()

print("Real-time text: ", end="", flush=True)
async for event in agent.stream_async(
    "Generate a weather forecast for Seattle: 68°F, partly cloudy, 55% humidity, 8 mph winds",
    structured_output_type=WeatherForecast
):
    if "data" in event:
        # Real-time text streaming
        print(event["data"], end="", flush=True)
    elif "result" in event:
        # Final structured output
        forecast = event["result"].structured_output
        print(f"\nStructured forecast: {forecast}")
```

### Using Tools with Structured Output

```python
from strands import Agent
from strands_tools import calculator
from pydantic import BaseModel, Field

class MathResult(BaseModel):
    operation: str = Field(description="the performed operation")
    result: int = Field(description="the result of the operation")

# Agent with tools and structured output
agent = Agent(tools=[calculator])
result = agent("What is 42 ^ 9", structured_output_type=MathResult)

print(result.structured_output.operation)  # "42^9"
print(result.structured_output.result)     # 406671383849472
```

### Agent with Default Structured Output

```python
# Set default output type for all calls
agent = Agent(structured_output_type=UserProfile)

# All calls will use UserProfile unless overridden
result1 = agent("Create a profile for John Doe, 25, dentist")
result2 = agent("Create a profile for Jane Smith, 30, teacher")

# Both results will have structured_output as UserProfile
```

### Session Management

```python
from uuid import uuid4
from strands.session.file_session_manager import FileSessionManager

session_id = str(uuid4())
session_manager = FileSessionManager(session_id=session_id)

agent = Agent(session_manager=session_manager)

# Conversation persists across calls
result1 = agent("Create profile for John", structured_output_type=UserProfile)
result2 = agent("What's his age?")  # Remembers John from previous call
```

## Use Cases

### 1. Data Extraction and Processing

```python
class ContactInfo(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None
    company: Optional[str] = None

# Extract structured data from unstructured text
agent = Agent()
result = agent(
    "Please extract: John Smith works at TechCorp, email john@techcorp.com, phone 555-1234",
    structured_output_type=ContactInfo
)
```

### 2. Task Management

```python
class TaskList(BaseModel):
    project_name: str
    tasks: List[str]
    priority: str = Field(pattern="^(high|medium|low)$")
    due_date: str
    estimated_hours: int

agent = Agent()
result = agent(
    "Create a project plan for website redesign: update homepage, fix mobile layout, "
    "optimize images, due next Friday, high priority, estimate 40 hours total",
    structured_output_type=TaskList
)
```

### 3. Content Analysis

```python
class SentimentAnalysis(BaseModel):
    text: str
    sentiment: str = Field(pattern="^(positive|negative|neutral)$")
    confidence: float = Field(ge=0.0, le=1.0)
    key_phrases: List[str]

agent = Agent()
result = agent(
    "Analyze: 'I love this product! It works perfectly and exceeded my expectations.'",
    structured_output_type=SentimentAnalysis
)
```

### 4. Multiple Output Types in One Agent

```python
from typing import Optional

class Person(BaseModel):
    name: str = Field(description="Full name")
    age: int = Field(description="Age in years", ge=0, le=150)
    email: str = Field(description="Email address")
    phone: Optional[str] = Field(description="Phone number", default=None)

class Task(BaseModel):
    title: str = Field(description="Task title")
    description: str = Field(description="Detailed description")
    priority: str = Field(description="Priority level: low, medium, high")
    completed: bool = Field(description="Whether task is completed", default=False)

agent = Agent()

# Different output types in same conversation
person_result = agent("Extract: John Doe, 35, john@test.com", structured_output_type=Person)
task_result = agent("Create task: Review code, high priority, completed", structured_output_type=Task)
```

## Best Practices

### 1. Model Design

- Use clear, descriptive field names
- Add helpful descriptions to fields
- Set appropriate default values
- Use validation constraints when needed

```python
class GoodModel(BaseModel):
    """Clear description of what this model represents."""
    user_name: str = Field(description="Full name of the user")
    age: int = Field(ge=0, le=150, description="Age in years")
    email: str = Field(description="Valid email address")
    is_active: bool = Field(default=True, description="Whether user is active")
```


### 2. Prompt Design

- Be specific about what data you want
- Provide examples when the format is complex
- Include all necessary context

```python
# Good prompt
result = agent(
    "Create a user profile for Sarah Johnson, age 28, software engineer at Google, "
    "currently active, lives in San Francisco",
    structured_output_type=UserProfile
)

# Less clear prompt
result = agent("Make a profile for Sarah", structured_output_type=UserProfile)
```

## Troubleshooting

### Common Issues

1. **Validation Errors**: The agent will automatically retry when validation fails
2. **Missing Required Fields**: Ensure your prompts include all necessary information
3. **Complex Nested Models**: Break down complex requests into smaller parts
4. **Tool Compatibility**: Structured output works with all existing Strands tools

### Debugging

```python
# Check if structured output was successful
if result.structured_output:
    print("Success:", result.structured_output)
else:
    print("No structured output generated")
    print("Message:", result.message)

# Check metrics for performance
print("Metrics:", result.metrics.get_summary())

# View conversation history
print("Messages:", agent.messages)
```

### Performance Tips

- Use simpler models for better performance
- Avoid overly complex nested structures when not needed
- Consider using streaming for real-time feedback
- Use session management for multi-turn conversations

## Integration with Observability

Structured output works seamlessly with observability platforms like Langfuse:

```python
import os
import base64
from datetime import datetime
from strands.telemetry import StrandsTelemetry

# Configure Langfuse
os.environ["LANGFUSE_PUBLIC_KEY"] = "your_key"
os.environ["LANGFUSE_SECRET_KEY"] = "your_secret"
os.environ["LANGFUSE_HOST"] = "https://us.cloud.langfuse.com"

# Setup telemetry
strands_telemetry = StrandsTelemetry().setup_otlp_exporter()

# Create agent with trace attributes
agent = Agent(
    trace_attributes={
        "session.id": f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "user.id": "user@example.com",
        "langfuse.tags": ["structured-output", "production"]
    }
)

result = agent("Create profile for John Doe, 25, engineer", structured_output_type=UserProfile)
```

---

This guide covers the core functionality of structured output with Strands Agents. For more advanced use cases and examples, refer to the test files and additional documentation.

## Appendix: Model Provider Code Examples

For quick reference to the model provider configuration examples shown in this document, use the links below:

- **Bedrock**: [View configuration example](#bedrock) ✅
- **Anthropic**: [View configuration example](#anthropic) ✅
- **LiteLLM**: [View configuration example](#litellm) ✅
- **Mistral**: [View configuration example](#mistral) ✅
- **OpenAI**: [View configuration example](#openai) ✅
- **Writer**: [View configuration example](#writer) ❌
- **Cohere**: [View configuration example](#cohere) ✅

Each section contains complete code examples showing how to configure and use that specific model provider with structured output, including authentication setup, parameter configuration, and usage patterns.


# Model Providers:
## Tested with the following model providers:

### Bedrock
```python

from strands import Agent
import os

agent = Agent(tools=[calculator])
response = agent("What is 2+2", MathResult)
print(response)
assert response.structured_output
response.structured_output
```


### Anthropic
```python

from strands import Agent
from strands.models.anthropic import AnthropicModel
import os


model = AnthropicModel(
    client_args={
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
    },
    max_tokens=1028,
    model_id="claude-sonnet-4-20250514",
    params={
        "temperature": 0.7,
    }
)

agent = Agent(model=model, tools=[calculator])
response = agent("What is 2+2", MathResult)
print(response)
assert response.structured_output
response.structured_output
```

### LiteLLM
```python
from strands import Agent
from strands.models.litellm import LiteLLMModel
from strands_tools import calculator

model = LiteLLMModel(
    client_args={
       "api_key": os.getenv("ANTHROPIC_API_KEY"),
    },
    # **model_config
    model_id="anthropic/claude-3-7-sonnet-20250219",
    params={
        "max_tokens": 1000,
        "temperature": 0.7,
    }
)

agent = Agent(model=model, tools=[calculator])
response = agent("What is 2+2", MathResult)
print(response)
assert response.structured_output
response.structured_output
```

### Mistral
```python
from strands import Agent
from strands.models.mistral import MistralModel
from strands_tools import calculator

model = MistralModel(
    api_key=os.getenv("MISTRAL_API_KEY"),
    # **model_config
    model_id="mistral-medium-2508",
)

agent = Agent(model=model, tools=[calculator])
response = agent("What is 2+2", MathResult)
print(response)
assert response.structured_output
response.structured_output
```

### OpenAI
```python
from strands import Agent
from strands.models.openai import OpenAIModel
from strands_tools import calculator

model = OpenAIModel(
    client_args={
        "api_key": os.getenv('OPENAI_API_KEY')
    },
    # **model_config
    model_id="gpt-4o",
    params={
        "max_tokens": 1000,
        "temperature": 0.7,
    }
)

agent = Agent(model=model, tools=[calculator])
response = agent("What is 2+2", MathResult)
print(response)
assert response.structured_output
response.structured_output
```

### Writer
```python
from strands import Agent
from strands.models.writer import WriterModel
from strands_tools import calculator

model = WriterModel(
    client_args={"api_key": os.getenv('WRITER_API_KEY')},
    # **model_config
    model_id="palmyra-x5",
)

agent = Agent(model=model, tools=[calculator])

response = agent("What is 2+2", MathResult)
print(response)
assert response.structured_output
response.structured_output
```


### Cohere
```python
from strands import Agent
from strands.models.openai import OpenAIModel
from strands_tools import calculator

model = OpenAIModel(
    client_args={
        "api_key": os.getenv('COHERE_API_KEY'),
        "base_url": "https://api.cohere.ai/compatibility/v1",  # Cohere compatibility endpoint
    },
    model_id="command-a-03-2025",  # or see https://docs.cohere.com/docs/models
    params={
        "stream_options": None
    }
)

response = agent("What is 2+2", MathResult)
print(response)
assert response.structured_output
response.structured_output
```
