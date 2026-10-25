import json
import os
import random
import re
from typing import Any

import networkx as nx
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

try:
    from groq import Groq
except ImportError:
    Groq = None

st.set_page_config(
    page_title="Smart City Dynamic Dispatch Grid",
    page_icon="🚑",
    layout="wide",
)

st.title("🚨 Smart City Dynamic Dispatch Grid")
st.caption("Synthetic emergency-call triage, incident prioritization, and vehicle dispatch demo")
st.warning(
    "Prototype only: this uses synthetic data and a simplified grid. "
    "Do not use it to make real emergency-response decisions."
)

# ----------------------- City and fleet -----------------------
GRID_SIZE = 20
LANDMARKS = {
    "Central Hospital": (3, 3),
    "City Hospital": (3, 3),
    "Railway Station": (15, 4),
    "Main Railway Station": (15, 4),
    "River Bridge": (10, 10),
    "Main Bridge": (10, 10),
    "Central Market": (6, 14),
    "City Market": (6, 14),
    "North School": (4, 17),
    "South School": (16, 15),
    "Fire Station": (2, 10),
    "Police Station": (17, 8),
    "Bus Terminal": (12, 3),
    "Petrol Pump": (8, 6),
    "Old Temple": (5, 5),
    "Shopping Mall": (13, 13),
    "River Bank": (10, 8),
    "Industrial Area": (18, 17),
    "Stadium": (11, 16),
    "Airport Road": (18, 3),
}

VEHICLES = [
    {"Vehicle ID": "AMB-01", "Type": "Ambulance", "Station": (2, 3)},
    {"Vehicle ID": "AMB-02", "Type": "Ambulance", "Station": (4, 12)},
    {"Vehicle ID": "AMB-03", "Type": "Ambulance", "Station": (14, 4)},
    {"Vehicle ID": "AMB-04", "Type": "Ambulance", "Station": (16, 15)},
    {"Vehicle ID": "AMB-05", "Type": "Ambulance", "Station": (8, 8)},
    {"Vehicle ID": "FIRE-01", "Type": "Fire Truck", "Station": (2, 10)},
    {"Vehicle ID": "FIRE-02", "Type": "Fire Truck", "Station": (17, 8)},
    {"Vehicle ID": "FIRE-03", "Type": "Fire Truck", "Station": (11, 16)},
    {"Vehicle ID": "RES-01", "Type": "Rescue Team", "Station": (10, 10)},
    {"Vehicle ID": "RES-02", "Type": "Rescue Team", "Station": (18, 17)},
]

def make_city_graph(seed: int = 7) -> nx.Graph:
    rng = random.Random(seed)
    graph = nx.grid_2d_graph(GRID_SIZE, GRID_SIZE)
    for u, v in graph.edges:
        graph.edges[u, v]["minutes"] = rng.randint(1, 3)
        graph.edges[u, v]["closed"] = False
    # River-like barrier with three crossing points, matching the demo's project concept.
    # Remove horizontal crossings between x=9 and x=10 except at bridge rows.
    bridge_rows = {3, 10, 16}
    for y in range(GRID_SIZE):
        if y not in bridge_rows:
            a, b = (9, y), (10, y)
            if graph.has_edge(a, b):
                graph.remove_edge(a, b)
    return graph

def add_closures(graph: nx.Graph, closure_rows: list[int], closure_cols: list[int]) -> nx.Graph:
    graph = graph.copy()
    # Close selected horizontal/vertical street segments, not entire junctions.
    for y in closure_rows:
        for x in range(GRID_SIZE - 1):
            if graph.has_edge((x, y), (x + 1, y)):
                graph.remove_edge((x, y), (x + 1, y))
    for x in closure_cols:
        for y in range(GRID_SIZE - 1):
            if graph.has_edge((x, y), (x, y + 1)):
                graph.remove_edge((x, y), (x, y + 1))
    return graph

def parse_json_response(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("The AI response did not contain a JSON object.")
    return json.loads(text[start:end + 1])

def rule_based_triage(call_text: str) -> dict[str, Any]:
    """Fallback triage if Groq is not configured or its request fails."""
    t = call_text.lower()
    if any(w in t for w in ["prank", "fake call", "nothing happened", "test call"]):
        return {
            "is_emergency": False, "incident_type": "Noise / prank",
            "severity": 1, "location": match_location(call_text),
            "summary": "Possible prank or non-emergency call", "confidence": 0.55
        }
    if any(w in t for w in ["fire", "smoke", "burning", "explosion"]):
        kind, severity = "Fire", 5
    elif any(w in t for w in ["trapped", "collapse", "building fell", "flood", "drowning"]):
        kind, severity = "Rescue", 5
    elif any(w in t for w in ["unconscious", "not breathing", "heart attack", "severe bleeding"]):
        kind, severity = "Medical", 5
    elif any(w in t for w in ["injured", "accident", "crash", "hurt", "bleeding"]):
        kind, severity = "Medical", 4
    else:
        kind, severity = "Medical", 3
    return {
        "is_emergency": True, "incident_type": kind, "severity": severity,
        "location": match_location(call_text),
        "summary": call_text.strip()[:180],
        "confidence": 0.55
    }

def match_location(text: str) -> str:
    lower = text.lower()
    for name in sorted(LANDMARKS, key=len, reverse=True):
        if name.lower() in lower:
            return name
    aliases = {
        "hospital": "Central Hospital",
        "railway": "Railway Station",
        "bridge": "River Bridge",
        "market": "Central Market",
        "school": "North School",
        "mall": "Shopping Mall",
        "stadium": "Stadium",
        "airport": "Airport Road",
        "temple": "Old Temple",
        "petrol": "Petrol Pump",
        "industrial": "Industrial Area",
        "bus stand": "Bus Terminal",
    }
    for phrase, landmark in aliases.items():
        if phrase in lower:
            return landmark
    return "Central Market"

def groq_triage(call_text: str, api_key: str, model: str) -> dict[str, Any]:
    if Groq is None:
        raise RuntimeError("Groq package is not installed. Install requirements.txt.")
    client = Groq(api_key=api_key)
    prompt = f"""
You are a cautious triage assistant for a SYNTHETIC emergency dispatch demonstration.
Extract a single call into valid JSON only with keys:
is_emergency (boolean), incident_type (one of Medical, Fire, Rescue, Noise / prank),
severity (integer 1-5, where 5 is most urgent), location (one exact string from allowed locations),
summary (short string), confidence (number 0-1).
Allowed locations: {list(LANDMARKS.keys())}.
Use the best matching location from the list. If no location is clear, use "Central Market"
and lower confidence. Do not invent facts. Prank/test/non-emergency calls should have
is_emergency=false and incident_type="Noise / prank". Severity must be 1-5.
Call text: {call_text!r}
"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return only a valid JSON object. This is a demo, not real emergency advice."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=350,
        response_format={"type": "json_object"},
    )
    data = parse_json_response(response.choices[0].message.content or "")
    data["severity"] = max(1, min(5, int(data.get("severity", 3))))
    if data.get("location") not in LANDMARKS:
        data["location"] = match_location(call_text)
        data["confidence"] = min(float(data.get("confidence", 0.5)), 0.4)
    data["is_emergency"] = bool(data.get("is_emergency", True))
    data["incident_type"] = str(data.get("incident_type", "Medical"))
    data["summary"] = str(data.get("summary", call_text[:180]))
    data["confidence"] = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
    return data

def route_minutes(graph: nx.Graph, start: tuple[int, int], end: tuple[int, int]) -> float:
    # Weighted shortest path based on simulated travel minutes.
    return float(nx.shortest_path_length(graph, start, end, weight="minutes"))

def route_nodes(graph: nx.Graph, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    return nx.shortest_path(graph, start, end, weight="minutes")

def choose_vehicle(incident_type: str, location: str, graph: nx.Graph, available: list[dict]) -> tuple[dict | None, float | None]:
    required = {"Medical": "Ambulance", "Fire": "Fire Truck", "Rescue": "Rescue Team"}.get(incident_type)
    candidates = [v for v in available if v["Type"] == required]
    if not candidates:
        # Demo fallback: allow any vehicle if a matching type is unavailable.
        candidates = available
    if not candidates:
        return None, None
    target = LANDMARKS[location]
    scored = []
    for v in candidates:
        try:
            minutes = route_minutes(graph, v["Station"], target)
            scored.append((minutes, v))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
    if not scored:
        return None, None
    minutes, vehicle = min(scored, key=lambda item: item[0])
    return vehicle, minutes

def draw_map(incidents: list[dict], assignments: list[dict], closures_rows: list[int], closures_cols: list[int]):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(-0.5, GRID_SIZE - 0.5)
    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))
    ax.grid(True, alpha=0.18)
    ax.set_xlabel("Grid X")
    ax.set_ylabel("Grid Y")
    ax.set_title("Simulated city grid (not a real geographic map)")
    # Landmarks
    for name, (x, y) in LANDMARKS.items():
        ax.scatter(x, y, marker="s", s=35)
        ax.annotate(name, (x, y), fontsize=6, xytext=(3, 3), textcoords="offset points")
    # Vehicle stations
    for v in VEHICLES:
        x, y = v["Station"]
        ax.scatter(x, y, marker="^", s=48)
    # Incidents
    for inc in incidents:
        if inc.get("is_emergency"):
            x, y = LANDMARKS.get(inc.get("location"), LANDMARKS["Central Market"])
            ax.scatter(x, y, marker="X", s=90)
    for y in closures_rows:
        ax.axhline(y, linewidth=2, linestyle="--", alpha=0.55)
    for x in closures_cols:
        ax.axvline(x, linewidth=2, linestyle="--", alpha=0.55)
    ax.text(0.01, -0.13, "Squares = landmarks | Triangles = vehicle stations | X = emergency incidents | Dashed lines = simulated closures",
            transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    return fig

# ----------------------- Sidebar -----------------------
st.sidebar.header("Configuration")
use_groq = st.sidebar.toggle("Use Groq AI for triage", value=True)
groq_key = st.sidebar.text_input(
    "Groq API key",
    value=os.getenv("GROQ_API_KEY", ""),
    type="password",
    help="Or add GROQ_API_KEY to Streamlit Secrets. Do not commit your key to GitHub.",
)
groq_model = st.sidebar.text_input("Groq model", value="llama-3.3-70b-versatile")
st.sidebar.caption("If Groq is disabled or unavailable, rule-based triage is used.")
st.sidebar.divider()
st.sidebar.subheader("Simulated road closures")
closure_rows = st.sidebar.multiselect("Close horizontal street row(s)", options=list(range(GRID_SIZE)), default=[])
closure_cols = st.sidebar.multiselect("Close vertical street column(s)", options=list(range(GRID_SIZE)), default=[])

if "calls" not in st.session_state:
    st.session_state.calls = []
if "dispatches" not in st.session_state:
    st.session_state.dispatches = []
if "used_vehicle_ids" not in st.session_state:
    st.session_state.used_vehicle_ids = set()

tab1, tab2, tab3 = st.tabs(["📞 Triage a call", "🚑 Dispatch board", "🗺️ City map"])

with tab1:
    st.subheader("Enter an incoming call")
    sample = st.selectbox("Try a sample call", [
        "Caller reports a serious road accident with two injured people near Central Hospital.",
        "Smoke and flames are coming from a building near the Shopping Mall.",
        "A person is unconscious and not breathing near Railway Station.",
        "Several people are trapped after a building collapse in the Industrial Area.",
        "This is a prank call; there is no emergency.",
        "A bus crashed near River Bridge and people need help.",
    ])
    call_text = st.text_area("Call transcript", value=sample, height=110)
    if st.button("Triage call", type="primary"):
        if not call_text.strip():
            st.error("Enter a call transcript first.")
        else:
            triage = None
            if use_groq and groq_key:
                try:
                    triage = groq_triage(call_text, groq_key, groq_model)
                    triage["triage_method"] = "Groq AI"
                except Exception as exc:
                    st.warning(f"Groq triage failed; using rule-based fallback. Details: {exc}")
            if triage is None:
                triage = rule_based_triage(call_text)
                triage["triage_method"] = "Rule-based fallback"
            record = {
                "Call ID": f"CALL-{len(st.session_state.calls) + 1:03d}",
                "Call text": call_text.strip(),
                **triage,
            }
            st.session_state.calls.append(record)
            st.success("Call triaged.")
            st.json({
                "Call ID": record["Call ID"],
                "Emergency": record["is_emergency"],
                "Incident type": record["incident_type"],
                "Severity (5 = highest)": record["severity"],
                "Location": record["location"],
                "Summary": record["summary"],
                "Confidence": record["confidence"],
                "Triage method": record["triage_method"],
            })
    if st.session_state.calls:
        st.markdown("#### Triage history")
        hist = pd.DataFrame(st.session_state.calls)
        show_cols = ["Call ID", "incident_type", "severity", "location", "is_emergency", "triage_method", "confidence"]
        hist = hist.rename(columns={
            "incident_type": "Incident type", "severity": "Severity",
            "location": "Location", "is_emergency": "Emergency?",
            "triage_method": "Triage method", "confidence": "Confidence"
        })
        st.dataframe(hist[[c for c in ["Call ID", "Incident type", "Severity", "Location", "Emergency?", "Triage method", "Confidence"] if c in hist.columns]], use_container_width=True)

with tab2:
    st.subheader("Prioritized incident and dispatch board")
    emergency_calls = [c for c in st.session_state.calls if c.get("is_emergency")]
    # A simple duplicate grouping by exact normalized location + incident type.
    grouped = {}
    for call in emergency_calls:
        key = (call.get("location"), call.get("incident_type"))
        if key not in grouped:
            grouped[key] = {
                "Incident ID": f"INC-{len(grouped) + 1:03d}",
                "Location": call.get("location"),
                "Incident type": call.get("incident_type"),
                "Severity": int(call.get("severity", 3)),
                "Calls": 1,
                "Summary": call.get("summary", ""),
                "Call IDs": [call.get("Call ID")],
            }
        else:
            grouped[key]["Calls"] += 1
            grouped[key]["Severity"] = max(grouped[key]["Severity"], int(call.get("severity", 3)))
            grouped[key]["Call IDs"].append(call.get("Call ID"))
    incidents = sorted(grouped.values(), key=lambda i: (-i["Severity"], i["Incident ID"]))
    if not incidents:
        st.info("Triage an emergency call first to populate the dispatch board.")
    else:
        board = pd.DataFrame(incidents)
        st.dataframe(board[["Incident ID", "Location", "Incident type", "Severity", "Calls", "Summary"]], use_container_width=True)
        if st.button("Dispatch available vehicles", type="primary"):
            graph = add_closures(make_city_graph(), closure_rows, closure_cols)
            available = [v.copy() for v in VEHICLES if v["Vehicle ID"] not in st.session_state.used_vehicle_ids]
            results = []
            for inc in incidents:
                assigned, minutes = choose_vehicle(inc["Incident type"], inc["Location"], graph, available)
                if assigned is None:
                    results.append({
                        "Incident ID": inc["Incident ID"], "Location": inc["Location"],
                        "Severity": inc["Severity"], "Vehicle": "UNASSIGNED",
                        "Estimated travel (simulated min)": None, "Status": "No available reachable vehicle"
                    })
                    continue
                # Claim the vehicle exactly once for this dispatch batch.
                st.session_state.used_vehicle_ids.add(assigned["Vehicle ID"])
                available = [v for v in available if v["Vehicle ID"] != assigned["Vehicle ID"]]
                results.append({
                    "Incident ID": inc["Incident ID"], "Location": inc["Location"],
                    "Severity": inc["Severity"], "Vehicle": assigned["Vehicle ID"],
                    "Estimated travel (simulated min)": round(minutes, 1), "Status": "Assigned"
                })
            st.session_state.dispatches = results
            st.success("Dispatch batch completed. Assigned vehicles cannot be reused in this session.")
        if st.session_state.dispatches:
            st.markdown("#### Latest dispatch result")
            st.dataframe(pd.DataFrame(st.session_state.dispatches), use_container_width=True)
            st.download_button(
                "Download dispatch results (CSV)",
                pd.DataFrame(st.session_state.dispatches).to_csv(index=False).encode("utf-8"),
                file_name="dispatch_results.csv",
                mime="text/csv",
            )
    if st.button("Reset demo state"):
        st.session_state.calls = []
        st.session_state.dispatches = []
        st.session_state.used_vehicle_ids = set()
        st.rerun()

with tab3:
    st.subheader("Simulated city map")
    st.caption("A grid simulation only; it does not represent real roads, GPS, or live traffic.")
    graph = add_closures(make_city_graph(), closure_rows, closure_cols)
    fig = draw_map(st.session_state.calls, st.session_state.dispatches, closure_rows, closure_cols)
    st.pyplot(fig, use_container_width=True)
    plt_obj = None
    st.markdown("#### Landmark directory")
    st.dataframe(
        pd.DataFrame([{"Landmark": name, "Grid X": xy[0], "Grid Y": xy[1]} for name, xy in LANDMARKS.items()]),
        use_container_width=True,
    )

st.divider()
st.caption("Audit note: this demo displays triage method, confidence, call history, incident grouping, and simulated dispatch estimates. Verify all logic and safety requirements before any real-world use.")
