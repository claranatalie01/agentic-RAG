import streamlit as st
import requests
import json

# Optional: install streamlit-geolocation for real GPS
# pip install streamlit-geolocation
try:
    from streamlit_geolocation import st_geolocation
    HAS_GEOLOCATION = True
except ImportError:
    HAS_GEOLOCATION = False
    st.warning("Install 'streamlit-geolocation' for location‑aware features")

st.set_page_config(page_title="HKPL Library Assistant", page_icon="📚")
st.title("📚 HKPL Library Assistant")

# ------------------------------------------------------------
# Location handling
# ------------------------------------------------------------
lat, lon = None, None
if HAS_GEOLOCATION:
    location = st_geolocation()
    if location and "latitude" in location:
        lat = location["latitude"]
        lon = location["longitude"]
        st.sidebar.success(f"📍 Location: {lat:.4f}, {lon:.4f}")
    else:
        st.sidebar.info("Click 'Get Location' to enable location‑aware answers")
else:
    # Fallback: manual library code selection (for testing)
    library_code = st.sidebar.selectbox(
        "Select library (for testing without GPS)",
        ["", "HKCL", "STPL", "TSTPL"],
        index=0
    )

# ------------------------------------------------------------
# Chat history
# ------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------------------------------------------------
# User input
# ------------------------------------------------------------
if prompt := st.chat_input("Ask me about library hours, book loans, etc."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build request payload
    payload = {"input_string": prompt}
    if HAS_GEOLOCATION and lat and lon:
        payload["latitude"] = lat
        payload["longitude"] = lon
    elif not HAS_GEOLOCATION and library_code:
        payload["library_code"] = library_code

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        try:
            with requests.post("http://localhost:8001/chat/stream",
                               json=payload,
                               stream=True) as resp:
                event_type = None
                for line in resp.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith("event:"):
                            event_type = line[6:].strip()
                        elif line.startswith("data:") and event_type == "answer":
                            data = line[5:].strip()
                            full_response += data
                            response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
        except Exception as e:
            st.error(f"Error: {e}")
        st.session_state.messages.append({"role": "assistant", "content": full_response})