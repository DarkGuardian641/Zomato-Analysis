import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

sys.path.insert(0, 'scripts')
from predict import predict, LABEL_COLORS, LABEL_EMOJIS

# ---- Page config ----
st.set_page_config(
    page_title='Zomato Analyzer',
    page_icon='🍽️',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ---- Custom CSS ----
st.markdown("""
<style>
    .result-card {
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 1px solid rgba(0,0,0,0.08);
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .stProgress > div > div > div {
        border-radius: 4px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ---- Load static data for dropdowns ----
@st.cache_data
def load_options():
    df = pd.read_csv('data/processed/clean_zomato.csv')
    cuisines = sorted(df['cuisines'].dropna().unique().tolist())
    locations = sorted(df['location'].dropna().unique().tolist())
    rest_types = sorted(df['rest_type'].dropna().unique().tolist())
    listing_types = sorted(df['listed_in(type)'].dropna().unique().tolist())
    return cuisines, locations, rest_types, listing_types

@st.cache_data
def load_stats():
    df = pd.read_csv('data/processed/clean_zomato.csv')
    df['cost_num'] = df['approx_cost(for two people)'].astype(str).str.replace(',','').str.strip()
    df['cost_num'] = pd.to_numeric(df['cost_num'], errors='coerce').fillna(500)
    def label(r):
        if r < 3.5: return 'Negative'
        elif r < 4.0: return 'Moderate'
        else: return 'Positive'
    df['feedback'] = df['rate'].apply(label)
    return df

try:
    cuisines_list, locations_list, rest_types_list, listing_types_list = load_options()
except FileNotFoundError:
    st.error("data/processed/clean_zomato.csv not found. Please ensure the dataset exists.")
    st.stop()

# ---- Sidebar — Navigation ----
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/7/75/Zomato_logo.png", width=140)
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["Predict Feedback", "Dataset Insights"])

# ============================================================
# PAGE 1 — PREDICT FEEDBACK
# ============================================================
if page == "Predict Feedback":
    st.title("🍽️ Zomato Restaurant Analyzer")
    st.caption("Enter restaurant details below to predict customer feedback — Negative, Moderate, or Positive.")
    st.markdown("---")

    if not os.path.exists('model/analyzer_model.pkl'):
        st.warning("Model not trained yet. Run `python scripts/train.py` first, then refresh this page.")
        st.code("python scripts/train.py", language='bash')
        st.stop()

    # ---- Input form ----
    with st.form("prediction_form"):
        st.subheader("Restaurant Details")

        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input(
                "Restaurant name",
                placeholder="e.g. Spice Garden",
                help="Enter the name of the restaurant"
            )
            cuisine = st.selectbox(
                "Cuisine type",
                options=cuisines_list,
                help="Select the primary cuisine served"
            )
            rest_type = st.selectbox(
                "Restaurant type",
                options=rest_types_list,
                help="Type of restaurant (e.g. Casual Dining, Quick Bites, Cafe)"
            )
            location = st.selectbox(
                "Location",
                options=locations_list,
                help="Area in Bangalore"
            )

        with col2:
            listing_type = st.selectbox(
                "Listed in (type)",
                options=listing_types_list,
                help="Category this restaurant falls under (e.g. Delivery, Dine-out)"
            )
            online_order = st.radio(
                "Online ordering available?",
                options=["Yes", "No"],
                horizontal=True
            )
            book_table = st.radio(
                "Table booking available?",
                options=["Yes", "No"],
                horizontal=True
            )
            cost = st.slider(
                "Approximate cost for two people (₹)",
                min_value=50,
                max_value=6000,
                value=500,
                step=50,
                help="Estimated cost for two people in rupees"
            )
            votes = st.number_input(
                "Number of votes / reviews",
                min_value=0,
                max_value=20000,
                value=100,
                step=10,
                help="How many people have reviewed this restaurant"
            )

        st.markdown("---")
        review_text = st.text_area(
            "📝 Analyze Customer Review (Optional)",
            placeholder="e.g. The food was fantastic and the ambiance was great, but the service was a bit slow.",
            help="If provided, we will combine the sentiment of this review with the restaurant's overall details."
        )

        submitted = st.form_submit_button("Predict Feedback", use_container_width=True, type="primary")

    # ---- Run prediction ----
    if submitted:
        if not name.strip():
            st.warning("Please enter a restaurant name.")
        else:
            input_data = {
                'name': name,
                'cuisines': cuisine,
                'rest_type': rest_type,
                'location': location,
                'listed_in(type)': listing_type,
                'online_order': online_order,
                'book_table': book_table,
                'approx_cost(for two people)': cost,
                'votes': votes,
            }

            with st.spinner("Analysing feedback..."):
                try:
                    result = predict(input_data)
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    st.stop()

                if review_text.strip():
                    import nltk
                    try:
                        nltk.data.find('sentiment/vader_lexicon.zip')
                    except LookupError:
                        nltk.download('vader_lexicon', quiet=True)
                    
                    from nltk.sentiment import SentimentIntensityAnalyzer
                    sia = SentimentIntensityAnalyzer()
                    scores = sia.polarity_scores(review_text)
                    comp = scores['compound']
                    
                    vader_probs = {'Negative': 0.0, 'Moderate': 0.0, 'Positive': 0.0}
                    if comp >= 0.05:
                        vader_probs['Positive'] = comp * 100
                        vader_probs['Moderate'] = (1 - comp) * 100
                    elif comp <= -0.05:
                        vader_probs['Negative'] = abs(comp) * 100
                        vader_probs['Moderate'] = (1 - abs(comp)) * 100
                    else:
                        vader_probs['Moderate'] = 100.0
                        
                    # Blend probabilities (50/50 mix)
                    for k in result['probabilities']:
                        result['probabilities'][k] = (result['probabilities'][k] + vader_probs[k]) / 2.0
                        
                    # Normalize
                    total = sum(result['probabilities'].values())
                    for k in result['probabilities']:
                        result['probabilities'][k] = (result['probabilities'][k] / total) * 100.0
                        
                    # Update label based on combined probabilities
                    new_label = max(result['probabilities'], key=result['probabilities'].get)
                    
                    border_colors = {'Negative': '#E24B4A', 'Moderate': '#F9CB42', 'Positive': '#1D9E75'}
                    emojis = {'Negative': '😞', 'Moderate': '😐', 'Positive': '😊'}
                    desc = {
                        'Negative': 'This restaurant is likely to receive poor customer feedback. Common issues include slow delivery, food quality, or pricing concerns.',
                        'Moderate': 'This restaurant is likely to receive average feedback. Customers generally find it acceptable but nothing exceptional.',
                        'Positive': 'This restaurant is likely to receive excellent customer feedback. Customers are generally very satisfied.'
                    }
                    
                    result['label'] = new_label
                    result['emoji'] = emojis[new_label]
                    result['color'] = border_colors[new_label]
                    result['description'] = desc[new_label]
                    result['confidence'] = result['probabilities'][new_label]

            # ---- Result display ----
            st.markdown("---")
            if review_text.strip():
                st.subheader("Prediction Result (Including Review Sentiment)")
            else:
                st.subheader("Prediction Result")

            # Main result card
            label = result['label']
            emoji = result['emoji']
            confidence = result['confidence']
            color = result['color']
            description = result['description']

            bg_colors = {'Negative': '#fff5f5', 'Moderate': '#fffbf0', 'Positive': '#f0fff8'}
            border_colors = {'Negative': '#E24B4A', 'Moderate': '#F9CB42', 'Positive': '#1D9E75'}

            st.markdown(f"""
            <div class="result-card" style="background:{bg_colors[label]}; border-left: 5px solid {border_colors[label]};">
                <h2 style="margin:0; color:{border_colors[label]};">{emoji} {label} Feedback</h2>
                <p style="font-size:1.1rem; color:#555; margin-top:0.5rem;">{description}</p>
                <p style="font-size:0.95rem; color:#888; margin:0;">Confidence: <strong>{confidence:.1f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)

            # Input summary
            with st.expander("View input summary"):
                summary_items = [{'Field': k, 'Value': str(v)} for k, v in input_data.items()]
                if review_text.strip():
                    summary_items.append({'Field': 'review_text', 'Value': review_text.strip()})
                summary_df = pd.DataFrame(summary_items)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ============================================================
# PAGE 2 — DATASET INSIGHTS
# ============================================================
elif page == "Dataset Insights":
    st.title("📊 Dataset Insights")
    st.caption("Exploratory analysis of the Zomato Bangalore restaurant dataset")
    st.markdown("---")

    df = load_stats()

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total restaurants", f"{len(df):,}")
    col2.metric("Avg rating", f"{df['rate'].mean():.2f} / 5")
    col3.metric("Unique locations", f"{df['location'].nunique()}")
    col4.metric("Unique cuisines", f"{df['cuisines'].nunique():,}")

    st.markdown("---")

    # ---- Row 1: Feedback distribution + Rating histogram ----
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.subheader("Feedback class distribution")
        counts = df['feedback'].value_counts()[['Negative', 'Moderate', 'Positive']]
        colors = ['#E24B4A', '#F9CB42', '#1D9E75']
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(counts.index, counts.values, color=colors, alpha=0.9, width=0.5, edgecolor='white')
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 150,
                    f'{val:,}\n({val/len(df)*100:.1f}%)', ha='center', va='bottom', fontsize=10)
        ax.set_ylabel('Count')
        ax.set_title('Feedback distribution', fontweight='bold')
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with r1c2:
        st.subheader("Rating distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(df['rate'], bins=30, color='#E23744', edgecolor='white', alpha=0.85)
        ax.axvline(3.5, color='#F9CB42', linestyle='--', lw=2, label='Neg/Mod boundary (3.5)')
        ax.axvline(4.0, color='#1D9E75', linestyle='--', lw=2, label='Mod/Pos boundary (4.0)')
        ax.set_xlabel('Rating')
        ax.set_ylabel('Count')
        ax.set_title('Rating distribution with feedback boundaries', fontweight='bold')
        ax.legend(fontsize=9)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ---- Row 2: Top locations + Cuisine type ----
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.subheader("Top 10 locations by positive feedback %")
        loc_pos = df[df['feedback'] == 'Positive'].groupby('location').size()
        loc_total = df.groupby('location').size()
        loc_pct = (loc_pos / loc_total * 100).dropna().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.barh(loc_pct.index[::-1], loc_pct.values[::-1], color='#1D9E75', alpha=0.85)
        ax.set_xlabel('Positive %')
        ax.set_title('Locations with highest positive feedback', fontweight='bold')
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with r2c2:
        st.subheader("Feedback split by listing type")
        pivot = df.groupby(['listed_in(type)', 'feedback']).size().unstack(fill_value=0)
        pivot = pivot.reindex(columns=['Negative', 'Moderate', 'Positive'], fill_value=0)
        pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
        fig, ax = plt.subplots(figsize=(5, 4))
        pivot_pct.plot(kind='barh', ax=ax, color=['#E24B4A','#F9CB42','#1D9E75'],
                       stacked=True, alpha=0.9, edgecolor='white')
        ax.set_xlabel('Percentage')
        ax.set_title('Feedback by listing category', fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ---- Row 3: Online order vs feedback ----
    st.subheader("Online ordering vs feedback")
    r3c1, r3c2 = st.columns(2)

    with r3c1:
        online_pivot = df.groupby(['online_order', 'feedback']).size().unstack(fill_value=0)
        online_pivot = online_pivot.reindex(columns=['Negative', 'Moderate', 'Positive'], fill_value=0)
        online_pct = online_pivot.div(online_pivot.sum(axis=1), axis=0) * 100
        fig, ax = plt.subplots(figsize=(5, 3.5))
        online_pct.plot(kind='bar', ax=ax, color=['#E24B4A','#F9CB42','#1D9E75'],
                        alpha=0.9, edgecolor='white', width=0.6)
        ax.set_xlabel('Online order')
        ax.set_ylabel('Percentage')
        ax.set_title('Feedback by online order availability', fontweight='bold')
        ax.legend(fontsize=9)
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(axis='x', rotation=0)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with r3c2:
        cost_pivot = df.groupby(['feedback'])['cost_num'].mean().reindex(['Negative', 'Moderate', 'Positive'])
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar(cost_pivot.index, cost_pivot.values, color=['#E24B4A','#F9CB42','#1D9E75'],
               alpha=0.9, width=0.5, edgecolor='white')
        for i, v in enumerate(cost_pivot.values):
            ax.text(i, v + 5, f'₹{v:.0f}', ha='center', fontsize=11, fontweight='bold')
        ax.set_ylabel('Avg cost for two (₹)')
        ax.set_title('Average cost per feedback class', fontweight='bold')
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

