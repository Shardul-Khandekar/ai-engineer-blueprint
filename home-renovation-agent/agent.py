from google.adk.agents import LlmAgent
from google.adk.tools import google_search

# Helper tool agent to perform Google searches
search_agent = LlmAgent(
    name="SearchAgent",
    model="gemini-2.5-flash",
    description="Searches for renovation costs, contractors, materials, and design trends",
    instruction="Use google_search to find current renovation information, costs, materials, and trends. Be concise and cite sources.",
    tools=[google_search],
)


# Information agent for general enquiries
info_agent = LlmAgent(
    name="InfoAgent",
    model="gemini-2.5-flash",
    description="Handles general renovation questions and provides system information",
    instruction="""
        You are the Info Agent for the AI Home Renovation Planner.

        WHEN TO USE: The coordinator routes general questions and casual greetings to you.

        YOUR RESPONSE:
        - Keep it brief and helpful (2-4 sentences)
        - Explain the system helps with home renovations using visual AI
        - Mention capabilities: photo analysis, design planning, budget estimation, timeline coordination
        - Ask about their renovation project (which room, can they share photos?)

        EXAMPLE:
        "Hi! I'm your AI Home Renovation Planner. I can analyze photos of your current space and inspiration images to 
        create a personalized renovation plan with design suggestions, budget estimates, and timelines. Which room are 
        you thinking of renovating? Feel free to share photos if you have them!"

        Be enthusiastic about home improvement and helpful!
        """
    )


# Rendering agent for iterative refinements
rendering_editor = LlmAgent(
    name="RenderingEditor",
    model="gemini-2.5-flash",
    description="Edits existing renovation renderings based on user feedback",
    instruction="""
        You refine existing renovation renderings.

        **TASK**: User wants to modify an existing rendering (e.g., "make cabinets cream", "darker flooring").

        **CRITICAL**: Find the most recent rendering filename from conversation history!
        Look for: "Saved as artifact: [filename]" or "kitchen_modern_renovation_v1.png" type references.

        Use **edit_renovation_rendering** tool:

        Parameters:
        1. artifact_filename: The exact filename of the most recent rendering
        2. prompt: Very specific edit instruction (be detailed!)
        3. asset_name: Base name without _vX (e.g., "kitchen_modern_renovation")

        **Example:**
        User: "Make the cabinets cream instead of white"
        Last rendering: "kitchen_modern_renovation_v1.png"

        Call: edit_renovation_rendering(
        artifact_filename="kitchen_modern_renovation_v1.png",
        prompt="Change the kitchen cabinets from white to a soft cream color (Benjamin Moore Cream Silk OC-14). Keep all other elements exactly the same: flooring, countertops, backsplash, lighting, appliances, and layout.",
        asset_name="kitchen_modern_renovation"

        Be SPECIFIC in prompts - vague = poor results!

        After editing, briefly confirm the change.

        **IMPORTANT - DO NOT use markdown image syntax!**
        - Do NOT output `![image](filename.png)` or similar markdown image links
        - Simply confirm the edit was successful and mention the artifact is available in the artifacts panel
        """
)