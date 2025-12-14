"""Small Streamlit playground for teaching the core widgets."""

from __future__ import annotations

import numpy as np
import streamlit as st

st.set_page_config(page_title="Streamlit basics", page_icon="✨", layout="wide")
st.title("Streamlit basics playground")
st.write(
    "This mini-app highlights the most common Streamlit patterns: layout, widgets, "
    "callbacks, and live charts."
)

with st.sidebar:
    st.header("Sidebar controls")
    energy = st.slider("How energetic is today's session?", 0, 10, 6)
    mood = st.selectbox("Room mood", ["Curious", "Focused", "Sleepy", "Hyped"], index=1)
    st.write(f"Energy: {energy} ⚡️ · Mood: {mood}")

name = st.text_input("What's your name?", placeholder="Type and press enter")
note = st.text_area(
    "What was the most interesting idea today?",
    placeholder="Widgets, layout, or charts?",
    height=100,
)

if "notes" not in st.session_state:
    st.session_state.notes = []

cols = st.columns(2)
with cols[0]:
    st.metric("Total notes", len(st.session_state.notes))
with cols[1]:
    show_chart = st.toggle("Live chart", value=True)

if st.button("Add note", type="primary", disabled=not note.strip()):
    st.session_state.notes.append({"name": name or "Anonymous", "note": note.strip()})
    st.success("Saved! Add another insight if you like.")

if st.session_state.notes:
    st.subheader("Collected insights")
    st.table(st.session_state.notes)
else:
    st.info("Use the form above to add your first note.")

if show_chart:
    st.subheader("Random learning velocity")
    chart_data = np.cumsum(np.random.randn(20, 3), axis=0)
    st.line_chart(chart_data, height=260)

st.caption(
    "Try editing this file to experiment with new widgets. Streamlit hot-reloads whenever you save."
)
