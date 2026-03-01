"""
Pydantic + instructor for production structured output.

Demonstrates:
1. Pydantic model with field validators
2. instructor with OpenAI (automatic retry on validation failure)
3. instructor with Anthropic

Requires: OPENAI_API_KEY and/or ANTHROPIC_API_KEY environment variables.

From: https://dadops.dev/blog/structured-output-from-llms/
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional


class Person(BaseModel):
    name: str = Field(description="Full name of the person")
    age: int = Field(description="Age in years", ge=0, le=150)
    email: Optional[str] = Field(default=None, description="Email if mentioned")

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Name cannot be empty")
        return v.strip()


def instructor_openai():
    """instructor + OpenAI — Pydantic validation with automatic retry."""
    import instructor
    from openai import OpenAI

    client = instructor.from_openai(OpenAI())

    person = client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=Person,       # your Pydantic model
        max_retries=3,               # retry on validation failure
        messages=[
            {"role": "user", "content": "John Smith is 27 years old. Email: john@example.com"}
        ]
    )

    print(person)           # name='John Smith' age=27 email='john@example.com'
    print(person.model_dump())  # {"name": "John Smith", "age": 27, "email": "john@example.com"}
    return person


def instructor_anthropic():
    """instructor + Anthropic — same pattern, different provider."""
    import instructor
    from anthropic import Anthropic

    client = instructor.from_anthropic(Anthropic())

    person = client.messages.create(
        model="claude-sonnet-4-6",
        response_model=Person,
        max_retries=3,
        max_tokens=256,
        messages=[
            {"role": "user", "content": "John Smith is 27 years old. Email: john@example.com"}
        ]
    )

    print(f"Name: {person.name}")   # John Smith
    print(f"Age: {person.age}")     # 27
    return person


if __name__ == "__main__":
    print("=== Pydantic + instructor Demos ===\n")

    # Test Pydantic model directly
    print("1. Pydantic model validation:")
    try:
        p = Person(name="John Smith", age=27, email="john@example.com")
        print(f"   Valid: {p}")
    except Exception as e:
        print(f"   Error: {e}")

    try:
        p = Person(name="", age=27)
        print(f"   Should not reach here: {p}")
    except Exception as e:
        print(f"   Caught empty name: {type(e).__name__}")

    try:
        p = Person(name="John", age=-5)
        print(f"   Should not reach here: {p}")
    except Exception as e:
        print(f"   Caught negative age: {type(e).__name__}")

    print("\n2. instructor + OpenAI:")
    try:
        instructor_openai()
    except Exception as e:
        print(f"   SKIP: {e}")

    print("\n3. instructor + Anthropic:")
    try:
        instructor_anthropic()
    except Exception as e:
        print(f"   SKIP: {e}")
