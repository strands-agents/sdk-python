#!/usr/bin/env python3
"""
Test script for function-based tools
"""

# import logging
from typing import Optional

from pydantic import BaseModel, Field

from strands import Agent

# logging.getLogger("strands").setLevel(logging.DEBUG)
# logging.basicConfig(format="%(levelname)s | %(name)s | %(message)s", handlers=[logging.StreamHandler()])

prompt = "Jane Smith is 28 years old and lives at 123 Main St, Boston, MA 02108."


class Person(BaseModel):
    """A simple model for testing structured data extraction."""

    name: str = Field(description="The person's full name")
    age: int = Field(description="The person's age in years", ge=18)


class Address(BaseModel):
    """Address information for testing nested data structures."""

    street: str = Field(description="Street address")
    city: str = Field(description="City name")
    zip_code: str = Field(description="Postal code", alias="zipCode")


class PersonWithAddress(BaseModel):
    """A person with an address."""

    person: Person = Field(description="The person's information")
    address: Optional[Address] = Field(description="The person's address")


# Initialize agent with function tools
print("\n===== Input Prompt =====\n")
print(prompt)

print("\n===== Running the agent =====\n")
result = Agent().with_output(prompt, PersonWithAddress)

print("\n===== Output Result =====\n")
print(result.model_dump_json(indent=2))
