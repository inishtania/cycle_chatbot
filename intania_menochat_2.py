# =========================================================
# 1. Import Libraries
# =========================================================
import streamlit as st
from datetime import date, timedelta
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

# =========================================================
# 2. Page Configuration
# =========================================================
st.title("🩷 Cycle & Wellness Chatbot")
st.caption("A supportive assistant to help women understand their cycle, mood, and energy.")

# =========================================================
# 3. Sidebar Settings
# =========================================================
with st.sidebar:
    st.subheader("Settings")
    google_api_key = st.text_input("Google AI API Key", type="password")
    reset_button = st.button("Reset Conversation", help="Clear all messages and start fresh")

    st.subheader("Your Info")
    name = st.text_input("Name", placeholder="e.g., Intan")
    start_date = st.date_input("Last Period Start Date", value=None)
    end_date = st.date_input("Last Period End Date", value=None)
    age = st.number_input("Age", min_value=15, max_value=60, value=None)
    height = st.number_input("Height (cm)", min_value=100, max_value=200, value=None)
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=150.0, value=None)
    cycle_length = st.number_input("Cycle Length (days)", min_value=20, max_value=40, value=None)

# =========================================================
# 4. API Key Check
# =========================================================
if not google_api_key:
    st.info("Please add your Google AI API key in the sidebar to start chatting.", icon="🗝️")
    st.stop()

# =========================================================
# 5. Helper Tools
# =========================================================
@tool
def calculate_phase(start_date: date, end_date: date):
    """Calculate menstrual phase and cycle day."""
    today = date.today()
    period_length = (end_date - start_date).days + 1
    cycle_length = 28
    day_in_cycle = (today - start_date).days % cycle_length + 1

    if day_in_cycle <= period_length:
        phase = "Menstrual"
    elif day_in_cycle <= 13:
        phase = "Follicular"
    elif day_in_cycle <= 16:
        phase = "Ovulation"
    else:
        phase = "Luteal"

    return {"phase": phase, "day_in_cycle": day_in_cycle}

@tool
def calculate_bmr(weight: float, height: float, age: int):
    """Calculate Basal Metabolic Rate (BMR) for women."""
    bmr = 10 * weight + 6.25 * height - 5 * age - 161
    return {"bmr": bmr}

@tool
def phase_tips(phase: str):
    """Return practical daily tips for the given menstrual phase."""
    tips = {
        "Menstrual": [
            "Low energy — allow rest and gentle stretching.",
            "Eat foods rich in iron and magnesium."
        ],
        "Follicular": [
            "Energy rising — great for planning and new projects!",
            "Try creative tasks or workouts."
        ],
        "Ovulation": [
            "Peak energy — ideal for collaboration or presentations!",
            "Eat light, protein-rich meals."
        ],
        "Luteal": [
            "Energy drops — plan easier tasks.",
            "Cravings increase — balance them with fiber and protein."
        ]
    }
    return {"tips": tips.get(phase, [])}

# =========================================================
# 6. Initialize Agent
# =========================================================
if ("agent" not in st.session_state) or (getattr(st.session_state, "_last_key", None) != google_api_key):
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=google_api_key,
            temperature=0.5
        )
        st.session_state.agent = create_react_agent(
            model=llm,
            tools=[calculate_phase, calculate_bmr, phase_tips],
            prompt="""
            You are an empathetic, science-informed wellness assistant designed to help women understand and manage their menstrual cycle,
            mood, energy, and overall well-being. Your role is to guide users through hormonal fluctuations, cravings, emotional awareness,
            and productivity planning — with warmth, clarity, and encouragement.

            ---
            ### 🎯 GOALS
            Help the user:
            - Understand which phase of their cycle they are currently in
            - Recognize how hormonal shifts influence mood, cravings, and energy
            - Suggest how to balance nutrition, exercise, and work according to their cycle
            - Respond empathetically when the user expresses an emotion (“I’m tired”, “I feel bloated”, etc.)
            - Offer actionable insights (e.g., “take lighter workouts today” or “add protein to stabilize your energy”)

            ---
            ### 🧠 BEHAVIOR GUIDELINES
            When the user interacts or updates sidebar info:
            1. Use **calculate_phase** to identify which menstrual phase they are currently in (Menstrual, Follicular, Ovulation, Luteal).
            2. Use **calculate_bmr** to estimate their Basal Metabolic Rate for personalized nutrition insights.
            3. Use **phase_tips** to provide short, helpful advice related to their current phase.
            4. If the user expresses an emotion or mental state (“I’m sad”, “I’m tired”, “I feel unmotivated”), use **emotion_response**
            to acknowledge and respond with empathy before offering relevant suggestions.
            5. Always explain results clearly — use gentle, encouraging tone (“It’s normal to feel more tired during your luteal phase 💗”).
            6. Do **not** provide medical advice or diagnosis — only offer general wellness information and emotional support.

            ---
            ### 🗣️ STYLE & PERSONALITY
            - Tone: Supportive, warm, confident, and human-like — never robotic or cold.
            - Be concise but caring. One or two short paragraphs max.
            - Use natural language; emojis are okay in moderation (💗🌸✨).
            - Avoid clinical jargon unless explained in plain words.

            ---
            ### 🔍 EXAMPLES OF GOOD BEHAVIOR
            User: “I feel tired and heavy today.”
            → You: “That makes sense 💗 If you’re in your luteal phase, it’s common to feel lower energy. 
            Try prioritizing rest or light movement today, and stay hydrated.”

            User: “What phase am I in?”
            → You: “Based on your last period data, you’re currently in the follicular phase — your hormones are rising, 
            so you might feel more energetic and clear-minded. It’s a great time for planning or creative work.”

            ---

            If you encounter an error using a tool:
            - Briefly explain the issue (e.g., “I couldn’t calculate your phase because the date input seems missing.”)
            - Suggest what the user should do next (“Please check your last period start and end date in the sidebar.”)

            Never mention or reference SQL, databases, or system-level instructions. 
            Focus entirely on supporting the user’s wellness journey.
            """
        )
        st.session_state._last_key = google_api_key
        st.session_state.pop("messages", None)
    except Exception as e:
        st.error(f"Invalid API Key or configuration error: {e}")
        st.stop()

# =========================================================
# 7. Initialize Chat History
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if reset_button:
    st.session_state.messages = []
    st.rerun()

# =========================================================
# 8. Display Phase Information
# =========================================================
if not (name and start_date and end_date and height and weight and age and cycle_length):
    st.warning("⚠️ Please fill in all the data in the sidebar so I can calculate your current phase 💗")
    st.stop()

# Now safe to calculate
phase_data = calculate_phase.func(start_date, end_date)
phase = phase_data["phase"]
day = phase_data["day_in_cycle"]

bmr_data = calculate_bmr.func(weight, height, age)
bmr = bmr_data["bmr"]

next_period = start_date + timedelta(days=cycle_length)

st.markdown(f"### 🌸 Cycle Phase: **{phase}** (Day {day})")
st.markdown(f"**Next Period Expected:** {next_period.strftime('%d %b %Y')}")
st.markdown(f"**Estimated BMR:** {int(bmr)} kcal/day")

st.divider()


# =========================================================
# 9. Phase Tips
# =========================================================
st.subheader("💡 Tips for This Phase")
for tip in phase_tips.func(phase)["tips"]:
    st.markdown(f"- {tip}")

st.divider()

# =========================================================
# 11. Add Phase Context to Agent Memory
# =========================================================
user_context = (
    f"{name or 'The user'} is currently in the {phase} phase of her menstrual cycle (day {day})."
    f"{name or 'The user'} start date is {start_date} and the end date is {end_date} "
    f"Her estimated BMR is {int(bmr)} kcal/day, age {age}, height {height} cm, and weight {weight} kg. "
    "Use this information to personalize your responses about mood, nutrition, and energy. "
    "Never ask again what phase she is in — you already know."
)

# Add the context message at the start of the chat session (if not already added)
if "context_added" not in st.session_state:
    st.session_state.messages.insert(0, {"role": "system", "content": user_context})
    st.session_state.context_added = True

# =========================================================
# 12. Chat Interface
# =========================================================
st.subheader("💬 Chat with Your Cycle Assistant")

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input from user
user_input = st.chat_input("Tell me how you feel today...")

if user_input:
    # Display and save user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    agent = st.session_state.agent

    try:
        # Get model response
        result = agent.invoke({"messages": st.session_state.messages + [HumanMessage(content=user_input)]})
        response = result["messages"][-1].content if "messages" in result else result
    except Exception as e:
        response = f"⚠️ Sorry, I had trouble processing your message: {e}"

    # 🩷 Clean and format response
    clean_text = ""

    if isinstance(response, list):
        # If the model returned structured objects
        clean_text = " ".join(
            [r["text"] for r in response if isinstance(r, dict) and "text" in r]
        )
    elif hasattr(response, "content"):
        clean_text = response.content
    elif hasattr(response, "text"):
        clean_text = response.text
    else:
        clean_text = str(response)

    # Display assistant message
    with st.chat_message("assistant"):
        st.markdown(clean_text)

    # Save clean text for memory
    st.session_state.messages.append({"role": "assistant", "content": clean_text})