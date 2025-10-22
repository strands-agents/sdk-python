"""
Integration tests for structured output with agents across all model providers.
"""

import asyncio
import os
from typing import List, Optional
from unittest import SkipTest

import pytest
from pydantic import BaseModel, Field, field_validator

from strands import Agent
from strands.models.gemini import GeminiModel
from strands.tools import tool
from tests_integ.models.providers import (
    ProviderInfo,
    all_providers,
    cohere,
    llama,
    gemini
)


# gemini flash needs tool choice becuase it's too weak to invoke without and our implementation did not yet add toolChoice. Therefore, we're using pro here
all_providers.remove(gemini)
gemini = ProviderInfo(
    id="gemini",
    environment_variable="GOOGLE_API_KEY",
    factory=lambda: GeminiModel(
        client_args={"api_key": os.getenv("GOOGLE_API_KEY")},
        model_id="gemini-2.5-pro",
        params={"temperature": 0.7},
    ),
)
all_providers.append(gemini)        

def get_models():
    """Get all model providers for parameterized testing."""
    return [
        pytest.param(
            provider_info,
            id=provider_info.id,
            marks=provider_info.mark,
        )
        for provider_info in all_providers
    ]


@pytest.fixture(params=get_models())
def provider_info(request) -> ProviderInfo:
    """Fixture that provides each model provider."""
    return request.param


@pytest.fixture()
def skip_for(provider_info: ProviderInfo):
    """Fixture to skip tests for specific providers."""

    def skip_for_any_provider_in_list(providers: list[ProviderInfo], description: str):
        if provider_info in providers:
            raise SkipTest(f"Skipping test for {provider_info.id}: {description}")

    return skip_for_any_provider_in_list


@pytest.fixture()
def model(provider_info):
    """Create a model instance from the provider."""
    return provider_info.create_model()


# ========== Pydantic Models for Structured Output ==========


class UserProfile(BaseModel):
    """User profile with validation."""

    name: str = Field(..., min_length=1, max_length=100)
    age: int = Field(..., ge=0, le=150)
    email: str = Field(..., pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")
    interests: List[str] = Field(..., min_length=1, max_length=10)

    @field_validator("interests")
    @classmethod
    def validate_interests(cls, v):
        if len(v) > 10:
            raise ValueError("Too many interests")
        return [interest.strip().lower() for interest in v]


class ProductRecommendation(BaseModel):
    """Product recommendation with nested structure."""

    product_id: str
    name: str
    price: float = Field(..., gt=0, le=1000000)
    category: str
    rating: float = Field(..., ge=0, le=5)
    features: List[str]
    user_match_score: float = Field(..., ge=0, le=1)


class OrderSummary(BaseModel):
    """Order summary with optional fields."""

    order_id: str
    customer_name: str
    items: List[str]
    total_amount: float = Field(..., gt=0)
    discount_applied: Optional[float] = None
    shipping_address: Optional[str] = None
    special_instructions: Optional[str] = None


class AnalysisReport(BaseModel):
    """Complex analysis report."""

    title: str
    summary: str = Field(..., min_length=10, max_length=500)
    key_findings: List[str] = Field(..., min_length=1, max_length=5)
    confidence_score: float = Field(..., ge=0, le=1)
    recommendations: List[str]
    metadata: dict


# ========== Tool Definitions ==========


@tool
def get_user_data(user_id: str) -> dict:
    """Get user data for testing."""
    return {
        "user_id": user_id,
        "name": "Test User",
        "age": 30,
        "email": "test@example.com",
        "interests": ["technology", "music", "travel"],
    }


@tool
def search_products(category: str, max_price: float) -> List[dict]:
    """Search for products."""
    return [
        {
            "product_id": "PROD-001",
            "name": "Smart Watch",
            "price": 299.99,
            "category": category,
            "rating": 4.5,
            "features": ["GPS", "Heart Rate Monitor", "Water Resistant"],
        },
        {
            "product_id": "PROD-002",
            "name": "Wireless Headphones",
            "price": 199.99,
            "category": category,
            "rating": 4.2,
            "features": ["Noise Cancelling", "30hr Battery", "Bluetooth 5.0"],
        },
    ]


@tool
def calculate_order_total(items: List[str], discount_percent: float = 0) -> dict:
    """Calculate order total."""
    base_total = len(items) * 100.0  # Simple calculation for testing
    discount = base_total * (discount_percent / 100)
    return {
        "subtotal": base_total,
        "discount": discount,
        "total": base_total - discount,
    }


# ========== Test Classes ==========


class TestStructuredOutputBasic:
    """Basic structured output tests."""

    def test_simple_structured_output(self, skip_for, model):
        """Test basic structured output using the modern approach."""
        agent = Agent(model=model)

        result = agent(
            "Create a profile for Alice, 28 years old, alice@example.com, likes hiking and photography",
            structured_output_model=UserProfile,
        )

        assert result.structured_output is not None
        assert isinstance(result.structured_output, UserProfile)
        assert result.structured_output.name == "Alice"
        assert result.structured_output.age == 28
        assert result.structured_output.email == "alice@example.com"
        assert len(result.structured_output.interests) >= 2
        assert "hiking" in result.structured_output.interests or "photography" in result.structured_output.interests

    def test_structured_output_with_validation(self, skip_for, model):
        """Test structured output with field validation using the modern approach."""
        agent = Agent(model=model)

        result = agent(
            "Recommend a smartwatch priced at $299.99 with 4.5 star rating",
            structured_output_model=ProductRecommendation,
        )

        assert result.structured_output is not None
        assert isinstance(result.structured_output, ProductRecommendation)
        assert 0 < result.structured_output.price <= 1000000
        assert 0 <= result.structured_output.rating <= 5
        assert len(result.structured_output.features) > 0

    def test_structured_output_with_optional_fields(self, skip_for, model):
        """Test structured output with optional fields using the modern approach."""
        agent = Agent(model=model)

        result = agent(
            "Create order ORD-123 for John Doe with 3 items totaling $150",
            structured_output_model=OrderSummary,
        )

        assert result.structured_output is not None
        assert isinstance(result.structured_output, OrderSummary)
        assert result.structured_output.order_id == "ORD-123"
        assert result.structured_output.customer_name == "John Doe"
        assert len(result.structured_output.items) > 0
        assert result.structured_output.total_amount > 0


class TestStructuredOutputWithTools:
    """Test structured output with tool execution."""

    def test_agent_with_tools_and_structured_output(self, skip_for, model):
        """Test agent using tools and returning structured output."""
        agent = Agent(
            model=model,
            tools=[get_user_data, search_products],
        )

        # Use structured_output_model parameter in agent invocation
        result = agent(
            "Get user data for user-123 and create a profile",
            structured_output_model=UserProfile,
        )

        assert result.structured_output is not None
        assert isinstance(result.structured_output, UserProfile)
        assert hasattr(result.structured_output, "name")
        assert hasattr(result.structured_output, "email")

    def test_multi_tool_workflow_with_structured_output(self, skip_for, model):
        """Test multiple tool calls with structured output."""
        agent = Agent(
            model=model,
            tools=[search_products, calculate_order_total],
        )

        result = agent(
            "Search for electronics under $500 and create an order summary",
            structured_output_model=OrderSummary,
        )

        assert result.structured_output is not None
        assert isinstance(result.structured_output, OrderSummary)
        assert result.structured_output.total_amount > 0


class TestStructuredOutputAsync:
    """Test async operations with structured output."""

    @pytest.mark.asyncio
    async def test_async_structured_output(self, skip_for, model):
        """Test async agent with structured output."""
        agent = Agent(model=model)

        result = await agent.invoke_async(
            "Create profile for Bob, 35, bob@test.com, enjoys cooking and gardening",
            structured_output_model=UserProfile,
        )

        assert result.structured_output is not None
        assert isinstance(result.structured_output, UserProfile)
        assert result.structured_output.name == "Bob"
        assert result.structured_output.age == 35

    @pytest.mark.asyncio
    async def test_concurrent_structured_outputs(self, skip_for, model):
        """Test concurrent structured output generation."""
        agent = Agent(model=model)

        tasks = [
            agent.invoke_async(
                f"Create profile for User{i}, age {25 + i}, user{i}@test.com, likes sports",
                structured_output_model=UserProfile,
            )
            for i in range(3)
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.structured_output is not None
            assert isinstance(result.structured_output, UserProfile)
            assert f"User{i}" in result.structured_output.name or f"user{i}" in result.structured_output.email.lower()

    @pytest.mark.asyncio
    async def test_streaming_with_structured_output(self, skip_for, model):
        """Test streaming with structured output."""
        skip_for([cohere, llama], "Streaming with structured output not fully supported")

        agent = Agent(model=model)

        # Stream the response using the correct API
        events = []
        async for event in agent.stream_async(
            "Recommend a laptop for $999 with 4.8 rating",
            structured_output_model=ProductRecommendation,
        ):
            events.append(event)

        # Final result should be in the last event
        assert len(events) > 0

        # Look for the result event
        result_event = None
        for event in events:
            if "result" in event:
                result_event = event
                break

        if result_event:
            result = result_event["result"]
            if result.structured_output:
                assert isinstance(result.structured_output, ProductRecommendation)


class TestStructuredOutputEdgeCases:
    """Test edge cases with structured output."""

    def test_empty_lists_handling(self, skip_for, model):
        """Test handling of empty lists in structured output."""

        class ListModel(BaseModel):
            required_list: List[str]
            optional_list: Optional[List[str]] = None

        agent = Agent(model=model)

        result = agent(
            "Create with required_list containing ['item1', 'item2']",
            structured_output_model=ListModel,
        )

        assert result.structured_output is not None
        assert isinstance(result.structured_output.required_list, list)
        assert len(result.structured_output.required_list) > 0

    def test_switching_structured_output_models(self, skip_for, model):
        """Test switching between different structured output models dynamically."""
        agent = Agent(model=model)

        # First call with UserProfile
        result1 = agent(
            "User: Model Switch Test, 40, switch@test.com, interests: testing, automation",
            structured_output_model=UserProfile,
        )
        assert result1.structured_output is not None
        assert isinstance(result1.structured_output, UserProfile)
        assert result1.structured_output.name == "Model Switch Test"
        assert result1.structured_output.age == 40

        # Second call with ProductRecommendation - different model
        result2 = agent(
            "Product: Switch Product, prod_switch, $99.99, Test category, "
            "Switching test product, 4.0 rating, features: versatile, dynamic",
            structured_output_model=ProductRecommendation,
        )
        assert result2.structured_output is not None
        assert isinstance(result2.structured_output, ProductRecommendation)
        assert result2.structured_output.product_id == "prod_switch"
        assert result2.structured_output.price == 99.99

    def test_no_structured_output_fallback(self, skip_for, model):
        """Test agent works normally without structured output."""
        agent = Agent(model=model)

        result = agent("What is 2 + 2?")

        # Should not have structured output
        assert result.structured_output is None
        # Should have normal response
        assert result.message is not None
        assert "content" in result.message
        # The answer should be in the response somewhere
        assert "4" in str(result.message["content"])

    def test_nested_models(self, skip_for, model):
        """Test deeply nested structured output."""

        class Address(BaseModel):
            street: str
            city: str
            zip_code: str

        class Person(BaseModel):
            name: str
            address: Address

        agent = Agent(model=model)

        result = agent(
            "Create person John Smith at 123 Main St, Springfield, 12345",
            structured_output_model=Person,
        )

        assert result.structured_output is not None
        assert isinstance(result.structured_output, Person)
        assert isinstance(result.structured_output.address, Address)
        assert result.structured_output.address.zip_code == "12345"
