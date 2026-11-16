"""Minimal GenAI wrapper example.

1. Put your own key in credentials.py.
2. Run:  python genai_wrapper_example.py

This is text chat only. The same OpenAI-style call shape is used in grading,
where the submitted wrapper is replaced by the trusted server wrapper.
"""
from credentials import API_KEY
from genai_wrapper import OpenAI


def main():
    client = OpenAI(api_key=API_KEY)
    response = client.chat.completions.create(
        model="gpt-oss:120b",
        messages=[
            {"role": "user", "content": "Reply with exactly: wrapper works"},
        ],
        temperature=0,
        max_tokens=32,
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
