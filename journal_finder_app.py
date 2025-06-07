import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Journal Finder", layout="centered")
st.title("🔎 Journal Finder (SJR 2024)")
st.write("Paste your abstract below, and we’ll suggest relevant journals from the SJR 2024 database.")

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

journals = load_journals()

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
