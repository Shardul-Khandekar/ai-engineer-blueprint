gr.ChatInterface component is a special "stateful" component. This means Gradio automatically creates and manages this history list in its backend for each user session.

Gradio Frontend
The browser bundles two things, the new message that user sends and the entire conversation history so far

Gradio Backend
This is responsible to call the Python function. After the function provides an output it automatically updates its internal history list.

Gradio Frontend
Gradio sends this new, complete history back to your browser. The browser then re-draws the chat window to show the full conversation.


Gradio history
Gradio saves the history as a list of pairs.
Each pair is a small list containing [user_message, bot_message]

Example
[
  ["Hello", "Hi there!"],
  ["What is RAG?", "RAG stands for Retrieval-Augmented Generation."]
]

LangChain history
The langchain object needs to know who said what. It requires a single, flat list of special objects, alternating between HumanMessage and AIMessage.

Example
[
  HumanMessage(content="Hello"),
  AIMessage(content="Hi there!"),
  HumanMessage(content="What is RAG?"),
  AIMessage(content="RAG stands for Retrieval-Augmented Generation.")
]