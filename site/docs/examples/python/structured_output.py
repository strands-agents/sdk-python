#!/usr/bin/env python3
"""
Structured Output Example

This example demonstrates how to use structured output with Strands Agents to
get type-safe, validated responses using Pydantic models.
"""
from typing import List, Optional

from pydantic import BaseModel, Field
from strands import Agent


def basic_example():
    """Basic example extracting structured information from text."""
    print("\n--- Basic Example ---")

    class PersonInfo(BaseModel):
        name: str
        age: int
        occupation: str

    agent = Agent()
    result = agent(
        "John Smith is a 30-year-old software engineer",
        structured_output_model=PersonInfo,
    )

    print(f"Name: {result.structured_output.name}")      # "John Smith"
    print(f"Age: {result.structured_output.age}")        # 30
    print(f"Job: {result.structured_output.occupation}") # "software engineer"


def complex_nested_model_example():
    """Example handling complex nested data structures."""
    print("\n--- Complex Nested Model Example ---")

    class Address(BaseModel):
        street: str
        city: str
        country: str
        postal_code: Optional[str] = None

    class Contact(BaseModel):
        email: Optional[str] = None
        phone: Optional[str] = None

    class Person(BaseModel):
        """Complete person information."""
        name: str = Field(description="Full name of the person")
        age: int = Field(description="Age in years")
        address: Address = Field(description="Home address")
        contacts: List[Contact] = Field(default_factory=list, description="Contact methods")
        skills: List[str] = Field(default_factory=list, description="Professional skills")

    agent = Agent()
    result = agent(
        "Extract info: Jane Doe, a systems admin, 28, lives at 123 Main St, New York, USA. Email: jane@example.com",
        structured_output_model=Person,
    )

    print(f"Name: {result.structured_output.name}")                    # "Jane Doe"
    print(f"Age: {result.structured_output.age}")                      # 28
    print(f"Street: {result.structured_output.address.street}")        # "123 Main St"
    print(f"City: {result.structured_output.address.city}")            # "New York"
    print(f"Country: {result.structured_output.address.country}")      # "USA"
    print(f"Email: {result.structured_output.contacts[0].email}")      # "jane@example.com"
    print(f"Skills: {result.structured_output.skills}")                # ["systems admin"]


if __name__ == "__main__":
    print("Structured Output Examples\n")

    basic_example()
    complex_nested_model_example()

    print("\nExamples completed.")
