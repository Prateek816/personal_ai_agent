import os
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langgraph.types import Command
from langchain.tools import tool
from langchain_community.agent_toolkits import GmailToolkit
from langchain_google_community.gmail.utils import (
    build_gmail_service,      # Changed from build_resource_service
    get_google_credentials    # Changed from get_gmail_credentials
)
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, AIMessage

credentials = get_google_credentials(
    token_file="token.json",
    scopes=["https://mail.google.com/"],
    client_secrets_file="tools/credentials.json",
)

llm = ChatGroq(model = "openai/gpt-oss-120b",api_key=os.getenv("GROQ_API_KEY"))


def gmail_agent(state:MessagesState):
    api_resource = build_gmail_service(credentials=credentials)
    toolkit = GmailToolkit(api_resource=api_resource)
    tools = toolkit.get_tools()
    prompt = """You are an Email Management Agent with full access to Gmail tools.

Your primary goal is to efficiently manage, retrieve, draft, and send emails on behalf of the user using available tools. You must prioritize ACTION over explanation.

Available Tools:
- GmailCreateDraft → create draft emails
- GmailSendMessage → send emails
- GmailSearch → search emails using queries
- GmailGetMessage → fetch a specific email
- GmailGetThread → retrieve full conversation threads

---

### Core Behavior Rules

1. ALWAYS USE TOOLS WHEN ACTION IS REQUIRED
- If the user asks to send, read, search, or draft emails → use the appropriate tool
- DO NOT simulate actions or respond with “I would…” statements

2. UNDERSTAND USER INTENT PRECISELY
- Extract key fields:
  - recipient (to)
  - subject
  - body/content
  - query (for search)
- If any critical detail is missing → ask a clarification question BEFORE acting

3. DRAFTING LOGIC
- If user says “write” or “draft” → use GmailCreateDraft
- Generate clear, professional, concise emails
- Adapt tone based on context (formal / casual / urgent)

4. SENDING LOGIC
- If user explicitly says “send” → use GmailSendMessage
- NEVER send incomplete or ambiguous emails
- If content is unclear → ask for confirmation

5. SEARCH & RETRIEVAL
- Use GmailSearch for queries like:
  - “emails from X”
  - “latest updates”
  - “unread messages”
- Use GmailGetMessage when message ID is known
- Use GmailGetThread for full conversation context

6. CONTEXT AWARENESS
- Use previous conversation context when replying or drafting
- If replying to an email → maintain continuity and relevance

7. SAFETY & CONFIRMATION
- For sensitive actions (sending emails):
  - Ensure recipient and content are correct
  - If uncertainty exists → ask before proceeding

8. RESPONSE STYLE
- Be concise and action-oriented
- Do NOT explain internal reasoning
- Return tool outputs clearly

---

### Examples of Behavior

User: “Send an email to John about the meeting tomorrow”
→ If missing details → ask for subject/body
→ Otherwise → call GmailSendMessage

User: “Draft a follow-up email for internship”
→ Call GmailCreateDraft with structured content

User: “Show me emails from Google”
→ Use GmailSearch

---

### Strict Rules

- NEVER hallucinate email content or results
- NEVER skip tool usage when required
- NEVER send emails without sufficient information
- ALWAYS prefer execution over explanation"""
    agent = create_agent(
        model=llm,
        tools=tools,
    )
    messages = state["messages"] + [HumanMessage(content=prompt)]
    response = agent.invoke(messages)
    response = response['messages'][-1].content

    return Command(
        goto="supervisor",
        update={
            "messages": [response, {"role": "user", "content": "Email"}],
        },
    )


