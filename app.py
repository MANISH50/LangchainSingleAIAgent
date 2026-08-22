import os
import time
import certifi
import requests
import streamlit as st

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks import BaseCallbackHandler

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_react_agent, AgentExecutor


# ============================================================
# CONFIG
# ============================================================

os.environ["SSL_CERT_FILE"] = certifi.where()

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")

st.set_page_config(
    page_title="AI Weather Intelligence",
    page_icon="🌤️",
    layout="wide"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
"""
<style>

.stApp {
    background:
        radial-gradient(
            circle at top right,
            #20204a 0%,
            #0b0b12 35%,
            #050509 100%
        );

    color: #ffffff;
}

.main {
    background: transparent;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}


/* HEADER */

.weather-title {
    text-align: center;
    font-size: 48px;
    font-weight: 850;

    background:
        linear-gradient(
            90deg,
            #8b9cff,
            #c084fc,
            #67e8f9
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    margin-bottom: 5px;
}

.weather-subtitle {
    text-align: center;
    color: #9ca3af;
    font-size: 16px;
    margin-bottom: 35px;
}


/* SEARCH */

.search-card {
    background: rgba(20,20,32,0.90);
    border: 1px solid #29293d;
    border-radius: 22px;
    padding: 22px;

    box-shadow:
        0 15px 50px rgba(0,0,0,0.35);
}


/* INPUT */

.stTextInput > div > div > input {
    background: #11111b !important;
    color: white !important;
    border: 1px solid #34344d !important;
    border-radius: 12px !important;
}

.stTextInput label {
    color: #c7c9d1 !important;
}


/* BUTTON */

.stButton > button {
    border-radius: 12px;
    border: 1px solid #6366f1;

    background:
        linear-gradient(
            90deg,
            #6366f1,
            #8b5cf6
        );

    color: white;

    font-weight: 700;

    height: 48px;

    transition:
        transform .2s ease,
        box-shadow .2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);

    box-shadow:
        0 8px 25px rgba(99,102,241,.35);
}


/* WEATHER HERO */

.weather-hero {
    padding: 32px;

    border-radius: 26px;

    background:
        linear-gradient(
            135deg,
            #18182a,
            #242444
        );

    border: 1px solid #38385a;

    box-shadow:
        0 15px 50px rgba(0,0,0,.4);

    position: relative;
    overflow: hidden;
}

.weather-hero::after {
    content: "";

    position: absolute;

    width: 260px;
    height: 260px;

    border-radius: 50%;

    background:
        rgba(139,92,246,.10);

    right: -80px;
    top: -90px;
}

.location-name {
    font-size: 21px;
    font-weight: 700;
}

.capital-name {
    font-size: 15px;
    color: #a7a9b8;
    margin-top: 6px;
}

.weather-icon {
    font-size: 82px;

    animation:
        floatWeather 3s ease-in-out infinite;
}

@keyframes floatWeather {

    0% {
        transform: translateY(0);
    }

    50% {
        transform: translateY(-10px);
    }

    100% {
        transform: translateY(0);
    }

}

.temperature {
    font-size: 65px;
    font-weight: 850;
    line-height: 1;
}

.condition {
    font-size: 21px;
    margin-top: 10px;
    color: #e4e4ec;
}

.feels-like {
    color: #9295a7;
    font-size: 14px;
    margin-top: 8px;
}


/* METRICS */

.metric {
    background:
        linear-gradient(
            145deg,
            #151520,
            #0f0f18
        );

    border:
        1px solid #29293d;

    padding: 23px;

    border-radius: 18px;

    text-align: center;

    min-height: 125px;

    box-shadow:
        0 8px 30px rgba(0,0,0,.25);

    transition:
        transform .25s ease,
        border .25s ease;
}

.metric:hover {
    transform: translateY(-5px);
    border-color: #5d5d91;
}

.metric-icon {
    font-size: 32px;
}

.metric-value {
    font-size: 25px;
    font-weight: 800;
    margin-top: 7px;
}

.metric-label {
    color: #858898;
    font-size: 13px;
    margin-top: 5px;
}


/* AGENT TRACKER */

.agent-box {
    background:
        linear-gradient(
            145deg,
            #151520,
            #0d0d15
        );

    border:
        1px solid #29293d;

    padding: 24px;

    border-radius: 22px;

    box-shadow:
        0 12px 40px rgba(0,0,0,.35);

    margin-top: 20px;
    margin-bottom: 25px;
}

.agent-header {
    font-size: 21px;
    font-weight: 800;
    margin-bottom: 17px;
}

.agent-step {
    display: flex;
    align-items: center;

    padding: 13px;

    margin: 7px 0;

    border-radius: 13px;

    background: #11111a;

    border: 1px solid transparent;

    transition:
        all .35s ease;
}

.agent-active {
    background:
        linear-gradient(
            90deg,
            #17172c,
            #20182e
        );

    border:
        1px solid #514e91;

    box-shadow:
        0 0 20px rgba(99,102,241,.12);
}

.agent-icon {
    width: 42px;
    text-align: center;
    font-size: 21px;
}

.agent-name {
    flex: 1;
    font-size: 14px;
    font-weight: 650;
}

.agent-status {
    font-size: 12px;
    color: #686b7b;
}

.active-status {
    color: #9b9cff;

    animation:
        pulseStatus 1.2s infinite;
}

@keyframes pulseStatus {

    0% {
        opacity: .35;
    }

    50% {
        opacity: 1;
    }

    100% {
        opacity: .35;
    }

}


/* SPINNER */

.spinner {
    display: inline-block;

    animation:
        spinAgent 1s linear infinite;
}

@keyframes spinAgent {

    from {
        transform: rotate(0deg);
    }

    to {
        transform: rotate(360deg);
    }

}


/* PROGRESS */

.progress-container {
    width: 100%;

    height: 8px;

    background: #252534;

    border-radius: 20px;

    overflow: hidden;

    margin-top: 18px;
}

.progress-bar {
    height: 100%;

    border-radius: 20px;

    background:
        linear-gradient(
            90deg,
            #6366f1,
            #8b5cf6,
            #22d3ee
        );

    background-size: 200% 100%;

    animation:
        progressAnimation 2s linear infinite;

    transition:
        width .5s ease;
}

@keyframes progressAnimation {

    0% {
        background-position: 0% 50%;
    }

    100% {
        background-position: 200% 50%;
    }

}


/* RAIN */

.rain-section-title {
    font-size: 25px;
    font-weight: 800;
    margin-top: 30px;
    margin-bottom: 15px;
}

.rain-card {
    background:
        linear-gradient(
            145deg,
            #151520,
            #0d0d15
        );

    border:
        1px solid #29293d;

    padding: 28px;

    border-radius: 22px;

    text-align: center;

    box-shadow:
        0 10px 35px rgba(0,0,0,.3);
}

.rain-icon {
    font-size: 60px;

    animation:
        floatWeather 3s ease-in-out infinite;
}

.rain-title {
    font-size: 23px;
    font-weight: 800;
    margin-top: 8px;
}

.rain-text {
    color: #8e91a2;
    margin-top: 7px;
}

.rain-value {
    font-size: 15px;
    font-weight: 700;
    margin-top: 14px;
}


/* COMPLETED */

.completed-badge {
    display: inline-block;

    padding: 9px 20px;

    border-radius: 20px;

    background: #10261b;

    border: 1px solid #1e6b43;

    color: #57e39b;

    font-weight: 700;

    font-size: 13px;
}


/* EXPANDER */

.streamlit-expanderHeader {
    background: #11111a !important;
    color: #ddd !important;
    border-radius: 12px !important;
}

</style>
""",
unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
"""
<div class="weather-title">
    🌤️ AI Weather Intelligence
</div>

<div class="weather-subtitle">
    LangChain ReAct Agent • Tavily Search • WeatherStack
</div>
""",
unsafe_allow_html=True
)


# ============================================================
# VALIDATION
# ============================================================

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is missing in .env")
    st.stop()

if not TAVILY_API_KEY:
    st.error("TAVILY_API_KEY is missing in .env")
    st.stop()

if not WEATHERSTACK_API_KEY:
    st.error("WEATHERSTACK_API_KEY is missing in .env")
    st.stop()


# ============================================================
# WEATHER ICON
# ============================================================

def get_weather_icon(condition):

    condition = str(condition).lower()

    if "thunder" in condition or "storm" in condition:
        return "⛈️"

    if (
        "rain" in condition
        or "drizzle" in condition
        or "shower" in condition
    ):
        return "🌧️"

    if "snow" in condition:
        return "❄️"

    if (
        "fog" in condition
        or "mist" in condition
        or "haze" in condition
    ):
        return "🌫️"

    if "cloud" in condition:
        return "☁️"

    if (
        "sun" in condition
        or "clear" in condition
    ):
        return "☀️"

    return "🌤️"


# ============================================================
# AGENT STEPS
# ============================================================

AGENT_STEPS = [
    ("🧠", "Understanding location"),
    ("🔎", "Finding capital"),
    ("🌤️", "Fetching current weather"),
    ("🧠", "Analysing conditions"),
    ("🌧️", "Checking rain outlook"),
    ("✨", "Preparing final report")
]


# ============================================================
# TRACKER
# ============================================================

def render_tracker(
    placeholder,
    current_step,
    completed=False
):

    total = len(AGENT_STEPS)

    progress = (
        100
        if completed
        else int((current_step / total) * 100)
    )

    html = """
<div class="agent-box">

<div class="agent-header">
    🤖 Live Agent Activity
</div>
"""

    for index, (icon, name) in enumerate(AGENT_STEPS):

        if completed or index < current_step:

            status_icon = "✅"
            status = "Completed"
            active_class = ""

        elif index == current_step:

            status_icon = '<span class="spinner">⚙️</span>'
            status = "Processing..."
            active_class = "agent-active"

        else:

            status_icon = "○"
            status = "Waiting"
            active_class = ""

        status_class = (
            "active-status"
            if index == current_step and not completed
            else ""
        )

        html += f"""
<div class="agent-step {active_class}">

<div class="agent-icon">
    {status_icon}
</div>

<div class="agent-name">
    {icon} {name}
</div>

<div class="agent-status {status_class}">
    {status}
</div>

</div>
"""

    completion_text = (
        "100% Complete"
        if completed
        else f"{progress}% Complete"
    )

    html += f"""
<div class="progress-container">

<div
    class="progress-bar"
    style="width:{progress}%;">
</div>

</div>

<div style="
    text-align:right;
    margin-top:8px;
    font-size:12px;
    color:#737687;
">

{completion_text}

</div>

</div>
"""

    placeholder.markdown(
        html,
        unsafe_allow_html=True
    )


# ============================================================
# WEATHER TOOL
# ============================================================

@tool
def get_weather_data(city: str) -> str:
    """
    Fetch current weather information for a city.
    """

    try:

        url = (
            "https://api.weatherstack.com/current"
            f"?access_key={WEATHERSTACK_API_KEY}"
            f"&query={city}"
        )

        response = requests.get(
            url,
            timeout=15
        )

        data = response.json()

        if "current" not in data:

            return (
                f"Could not fetch weather for {city}. "
                f"API response: {data}"
            )

        current = data["current"]

        descriptions = current.get(
            "weather_descriptions",
            ["Unknown"]
        )

        weather_data = {
            "city": city,
            "temperature": current.get("temperature"),
            "feels_like": current.get("feelslike"),
            "condition": descriptions[0],
            "humidity": current.get("humidity"),
            "cloud_cover": current.get("cloudcover"),
            "wind_speed": current.get("wind_speed"),
            "wind_direction": current.get("wind_dir"),
            "visibility": current.get("visibility"),
            "pressure": current.get("pressure")
        }

        st.session_state["weather_data"] = weather_data

        return str(weather_data)

    except Exception as e:

        return f"Weather API error: {str(e)}"


# ============================================================
# TAVILY
# ============================================================

search_tool = TavilySearchResults(
    max_results=5
)


# ============================================================
# LLM
# ============================================================

llm = ChatOpenAI(
    model="gpt-4.1-nano-2025-04-14",
    temperature=0,
    api_key=OPENAI_API_KEY
)


# ============================================================
# PROMPT
# ============================================================

prompt = PromptTemplate.from_template(
"""
You are an intelligent Weather Intelligence Agent.

Available tools:

{tools}

Tool names:

{tool_names}


TASK:

The user provides a city or state.

You must:

1. Understand the location.
2. Determine its country/state if required.
3. Find the relevant capital using Tavily.
4. Fetch current weather using WeatherStack.
5. Analyse current conditions.
6. Assess the possibility of rain during the next hour.

IMPORTANT:

WeatherStack CURRENT API provides current weather.

It may not provide genuine next-hour precipitation
probability.

Never invent an exact rain probability.

If hourly forecast data is unavailable,
clearly say that the rain outlook is an estimate
based on current atmospheric conditions.


Use:

Question: user question

Thought: reasoning

Action: one of [{tool_names}]

Action Input: tool input

Observation: tool result

Thought: final reasoning

Final Answer: final answer


Question: {input}

Thought:{agent_scratchpad}
"""
)


# ============================================================
# AGENT
# ============================================================

tools = [
    search_tool,
    get_weather_data
]

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)


# ============================================================
# CALLBACK
# ============================================================

class WeatherAgentCallback(BaseCallbackHandler):

    def __init__(self, tracker):

        self.tracker = tracker
        self.current_step = 0

    def update(self, step):

        self.current_step = step

        render_tracker(
            self.tracker,
            step
        )

    def on_chain_start(
        self,
        serialized,
        inputs,
        **kwargs
    ):

        self.update(0)

    def on_tool_start(
        self,
        serialized,
        input_str,
        **kwargs
    ):

        tool_name = ""

        if serialized:

            tool_name = str(
                serialized.get(
                    "name",
                    ""
                )
            )

        if "tavily" in tool_name.lower():

            self.update(1)

        elif "weather" in tool_name.lower():

            self.update(2)

        else:

            self.update(3)

    def on_tool_end(
        self,
        output,
        **kwargs
    ):

        if self.current_step < 5:

            self.update(
                self.current_step + 1
            )


# ============================================================
# WEATHER DISPLAY
# ============================================================

def display_weather(
    weather,
    capital
):

    city = weather.get(
        "city",
        "Unknown"
    )

    temperature = weather.get(
        "temperature",
        "--"
    )

    feels_like = weather.get(
        "feels_like",
        "--"
    )

    condition = weather.get(
        "condition",
        "Unknown"
    )

    humidity = weather.get(
        "humidity",
        "--"
    )

    cloud = weather.get(
        "cloud_cover",
        "--"
    )

    wind = weather.get(
        "wind_speed",
        "--"
    )

    icon = get_weather_icon(
        condition
    )

    hero_html = f"""
<div class="weather-hero">

<div class="location-name">
    📍 {city}
</div>

<div class="capital-name">
    🏛️ Capital: {capital}
</div>

<div style="
    display:flex;
    align-items:center;
    margin-top:25px;
    position:relative;
    z-index:2;
">

<div class="weather-icon">
    {icon}
</div>

<div style="margin-left:25px;">

<div class="temperature">
    {temperature}°C
</div>

<div class="condition">
    {condition}
</div>

<div class="feels-like">
    Feels like {feels_like}°C
</div>

</div>

</div>

</div>
"""

    st.markdown(
        hero_html,
        unsafe_allow_html=True
    )

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    metrics = [
        (col1, "💧", f"{humidity}%", "Humidity"),
        (col2, "☁️", f"{cloud}%", "Cloud Cover"),
        (col3, "💨", f"{wind} km/h", "Wind Speed"),
        (col4, "🌡️", f"{feels_like}°C", "Feels Like")
    ]

    for column, icon, value, label in metrics:

        metric_html = f"""
<div class="metric">

<div class="metric-icon">
    {icon}
</div>

<div class="metric-value">
    {value}
</div>

<div class="metric-label">
    {label}
</div>

</div>
"""

        with column:

            st.markdown(
                metric_html,
                unsafe_allow_html=True
            )


# ============================================================
# RAIN OUTLOOK
# ============================================================

def display_rain_outlook(weather):

    condition = str(
        weather.get(
            "condition",
            ""
        )
    ).lower()

    humidity = weather.get(
        "humidity",
        0
    )

    cloud = weather.get(
        "cloud_cover",
        0
    )

    score = 0

    if (
        "rain" in condition
        or "drizzle" in condition
        or "shower" in condition
    ):

        score += 60

    if isinstance(cloud, (int, float)):

        if cloud >= 80:
            score += 25

        elif cloud >= 60:
            score += 15

        elif cloud >= 40:
            score += 8

    if isinstance(humidity, (int, float)):

        if humidity >= 85:
            score += 15

        elif humidity >= 70:
            score += 8

    score = min(score, 95)

    if score >= 65:

        icon = "🌧️"
        title = "High Rain Signal"
        description = (
            "Current conditions show strong "
            "signals associated with rain."
        )

    elif score >= 35:

        icon = "🌦️"
        title = "Moderate Rain Signal"
        description = (
            "Current conditions show some "
            "potential for rain."
        )

    else:

        icon = "🌤️"
        title = "Low Rain Signal"
        description = (
            "Current conditions show relatively "
            "weak signals for immediate rain."
        )

    st.markdown(
        """
<div class="rain-section-title">
    🌧️ Rain Outlook
</div>
""",
        unsafe_allow_html=True
    )

    rain_html = f"""
<div class="rain-card">

<div class="rain-icon">
    {icon}
</div>

<div class="rain-title">
    {title}
</div>

<div class="rain-text">
    {description}
</div>

<div class="progress-container">

<div
    class="progress-bar"
    style="width:{score}%;">
</div>

</div>

<div class="rain-value">
    Atmospheric Signal: {score}%
</div>

<div style="
    margin-top:9px;
    font-size:12px;
    color:#686b7b;
">

Estimated from current atmospheric conditions.

</div>

</div>
"""

    st.markdown(
        rain_html,
        unsafe_allow_html=True
    )


# ============================================================
# SEARCH UI
# ============================================================

st.markdown(
"""
<div class="search-card">

<div style="
    font-size:21px;
    font-weight:750;
    margin-bottom:10px;
">

📍 Search Location

</div>

<div style="
    color:#777b8c;
    font-size:13px;
">

Enter a city, state or location.

</div>

</div>
""",
unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)


location = st.text_input(
    "Location",
    placeholder="Example: Hyderabad, Mumbai, Kashmir...",
    label_visibility="collapsed"
)

analyze = st.button(
    "🚀 Analyze Weather",
    type="primary",
    use_container_width=True
)


# ============================================================
# EXECUTION
# ============================================================

if analyze:

    if not location.strip():

        st.warning(
            "Please enter a city or state."
        )

        st.stop()

    st.session_state["weather_data"] = None

    tracker = st.empty()

    render_tracker(
        tracker,
        0
    )

    start_time = time.perf_counter()

    callback = WeatherAgentCallback(
        tracker
    )

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        callbacks=[callback]
    )

    query = f"""
The user entered:

{location}

Find the appropriate capital.

Fetch the current weather.

Analyse the current weather.

Assess the possibility of rain during
the next hour.

Do not invent a precise rain probability
if hourly forecast information is unavailable.
"""

    try:

        result = executor.invoke(
            {
                "input": query
            }
        )

        render_tracker(
            tracker,
            6,
            completed=True
        )

        execution_time = (
            time.perf_counter()
            - start_time
        )

        weather = st.session_state.get(
            "weather_data"
        )

        if not weather:

            st.error(
                "Weather data was not returned."
            )

            st.stop()

        final_answer = result.get(
            "output",
            ""
        )

        # ----------------------------------------------------
        # CAPITAL EXTRACTION
        # ----------------------------------------------------

        capital = location

        for line in final_answer.split("\n"):

            lower = line.lower()

            if "capital:" in lower:

                capital = line.split(
                    ":",
                    1
                )[1].strip()

                break

            if "capital is" in lower:

                capital = line.split(
                    "capital is",
                    1
                )[1].strip()

                break

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.markdown("---")

        st.markdown(
            """
<div style="
    font-size:28px;
    font-weight:800;
    margin-bottom:18px;
">

🌦️ Weather Intelligence Report

</div>
""",
            unsafe_allow_html=True
        )

        display_weather(
            weather,
            capital
        )

        display_rain_outlook(
            weather
        )

        st.markdown(
            f"""
<div style="
    text-align:center;
    margin-top:28px;
">

<span class="completed-badge">

✅ Agent completed
&nbsp; • &nbsp;
{execution_time:.2f}s

</span>

</div>
""",
            unsafe_allow_html=True
        )

        with st.expander(
            "🧠 View Agent Response"
        ):

            st.write(
                final_answer
            )

    except Exception as e:

        st.error(
            "❌ Agent execution failed"
        )

        st.exception(e)