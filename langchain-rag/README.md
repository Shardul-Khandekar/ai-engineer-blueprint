gr.ChatInterface component is a special "stateful" component. This means Gradio automatically creates and manages this history list in its backend for each user session.
**LangChain + Gradio — Chat history formats and conversion**

A short guide showing how Gradio manages chat history and how to convert that history into the format LangChain expects (and back). This is useful when building a Retrieval-Augmented Generation (RAG) chat app where Gradio is the UI and LangChain is used for message handling and model orchestration.

**Gradio: stateful chat history**

The `gr.ChatInterface` component (or `gr.Chatbot`/`gr.Chat` depending on your version) is a stateful component. Gradio maintains a per-session history as a list of pairs where each pair is `[user_message, bot_message]`.

Example (Gradio history):

```py
[
  ["Hello", "Hi there!"],
  ["What is RAG?", "RAG stands for Retrieval-Augmented Generation."]
]
```

Flow (high level):
- Frontend bundles the new user message and the full conversation history and sends them to the backend handler.
- The backend function processes the input, produces a response, and Gradio updates its internal history for the session.
- Gradio returns the updated history to the browser and the chat window is re-drawn.

**LangChain: flat message list**

LangChain expects a flat, ordered list of message objects that preserve speaker identity. These objects typically alternate between `HumanMessage` and `AIMessage`.

Example (LangChain history):

```py
[ 
  HumanMessage(content="Hello"),
  AIMessage(content="Hi there!"),
  HumanMessage(content="What is RAG?"),
  AIMessage(content="RAG stands for Retrieval-Augmented Generation.")
]
```