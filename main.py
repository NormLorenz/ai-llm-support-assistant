import os
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import google.generativeai
import gradio as gr


# Load environment variables in a file called .env
load_dotenv(override=True)

openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')


# Connect to OpenAI, Anthropic and Google
openai = OpenAI()
claude = anthropic.Anthropic()
google.generativeai.configure()


# A generic system message
system_message = "You are a helpful assistant in a clothes store. You should try to gently encourage \
the customer to try items that are on sale. Hats are 60% off, and most other items are 50% off. \
For example, if the customer says 'I'm looking to buy a hat', \
you could reply something like, 'Wonderful - we have lots of hats - including several that are part of our sales event.'\
Encourage the customer to buy hats if they are unsure what to get."

# Default model
MODEL = "gpt-4o-mini"


def main() -> None:
    ### High level entry point ###
    gr.ChatInterface(fn=chat, type="messages").launch(inbrowser=True)


def chat(message, history):
    _system_message = system_message

    # Add specific instructions based on user input
    if 'belt' in message:
        _system_message += " The store does not sell belts; if you are asked for belts, be sure to point out other items on sale."

    # Prepare messages for OpenAI
    messages = [{"role": "system", "content": _system_message}] + \
        history + [{"role": "user", "content": message}]

    # Call the OpenAI API with streaming
    stream = openai.chat.completions.create(
        model=MODEL, messages=messages, stream=True)

    # Stream the response back to the user
    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        yield response

# def select_model(company_name: str, url: str, model: str):
#     yield ""
#     prompt = f"Please generate a company brochure for {company_name}. Here is their landing page:\n"
#     prompt += Website(url).get_contents()
#     if model == "gpt-4o-mini":
#         result = call_gpt(prompt, model)
#     elif model == "claude-3-5-haiku-latest":
#         result = call_claude(prompt, model)
#     elif model == "gemini-1.5-flash":
#         result = call_gemini(prompt, model)
#     elif model == "gemini-2.5-flash-lite":
#         result = call_gemini(prompt, model)
#     else:
#         raise ValueError("Unknown model")
#     yield from result


def call_gpt(prompt: str, model: str):
    # Prepare messages for OpenAI
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]

    # Call the OpenAI API with streaming
    stream = openai.chat.completions.create(
        model=model,
        messages=messages,
        stream=True
    )

    # Stream the response back to the user
    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ""
        yield response


def call_claude(prompt: str, model: str):
    # Prepare messages for Claude
    messages = [{"role": "user", "content": prompt}]

    # Call the Claude API with streaming
    result = claude.messages.stream(
        model=model,
        system=system_message,
        messages=messages,
        temperature=0.7,
        max_tokens=1000
    )

    # Stream the response back to the user
    response = ""
    with result as stream:
        for text in stream.text_stream:
            response += text or ""
            yield response


def call_gemini(prompt: str, model: str):
    # Call the Gemini API with streaming
    gemini = google.generativeai.GenerativeModel(
        model_name=model,
        system_instruction=system_message
    )

    # Stream the response back to the user
    result = ""
    for response in gemini.generate_content(prompt, stream=True):
        result += response.text or ""
        yield result


if __name__ == "__main__":
    main()
