import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ✅ Must be the first Streamlit command
st.set_page_config(page_title="Journal Finder", layout="centered")

# 🔷 Logo and Branding
st.image("ResearchMate1.png", width=180)
st.markdown("### Developed by **Abdollah Baghaei Daemei** – [ResearchMate.org](https://www.researchmate.org)")
st.markdown("---")

# 📚 Load Journal Data
@st.cache_data
def load_journals():
    return pd.read_csv("sjr_2024.csv", sep=';')

def find_matches(abstract, journals_df, top_n=10):
    corpus = journals_df['Title'].tolist() + [abstract]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    journals_df['Score'] = similarities
    return journals_df.sort_values(by="Score", ascending=False).head(top_n)

# 🧠 Main App Interface
journals = load_journals()

st.title("🔎 Journal Finder (SJR 2024)")
st.write("Paste your abstract below, and we’ll suggest relevant journals from the SJR 2024 database.")

abstract = st.text_area("✍️ Enter your abstract here", height=250)

if st.button("Find Journals") and abstract.strip():
    results = find_matches(abstract, journals)
    st.success("🎯 Top Matching Journals:")
    for idx, row in results.iterrows():
        st.markdown(f"""**{row['Title']}**  
📊 *SJR*: {row['SJR']} | 🏅 *Quartile*: {row['SJR Best Quartile']}  
📚 *Categories*: {row['Categories']}  
🌍 *Country*: {row['Country']} | 🏢 *Publisher*: {row['Publisher']}  
---""")

# 📥 Download Buttons
with open("README.md", "r", encoding="utf-8") as f:
    readme_text = f.read()

with open("LICENSE", "r", encoding="utf-8") as f:
    license_text = f.read()

st.download_button(
    label="📘 Download README",
    data=readme_text,
    file_name="README.md",
    mime="text/markdown"
)
st.download_button(
    label="📜 Download LICENSE",
    data=license_text,
    file_name="LICENSE",
    mime="text/plain"
)
