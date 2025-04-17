import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageOps
import os

st.title("ESE 3 Project: Data & Image Insight Analyzer")

# ---------------------
# TABS FOR ORGANIZATION
# ---------------------
tab1, tab2 = st.tabs(["Data Analysis", "Image Processing"])

# ---------------------
# PART 1: DATA ANALYSIS
# ---------------------
with tab1:
    st.header("Part 1: Titanic Dataset Analysis")

    # Load Dataset
    df = pd.read_csv(r"E:\Codes\Advanced Python\Dataset\csv\synthetic_titanic.csv")

    # Interactive Filters
    st.sidebar.header("Data Filters")
    sex_filter = st.sidebar.selectbox("Filter by Sex", options=['All'] + list(df['Sex'].unique()))
    age_range = st.sidebar.slider("Select Age Range", int(df['Age'].min()), int(df['Age'].max()), (10, 60))

    filtered_df = df.copy()
    if sex_filter != 'All':
        filtered_df = filtered_df[filtered_df['Sex'] == sex_filter]
    filtered_df = filtered_df[filtered_df['Age'].between(*age_range)]

    if st.sidebar.checkbox("Show raw data"):
        st.write(filtered_df.head())

    # Visualizations
    st.subheader("1. Survival Count")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=filtered_df, x='Survived', ax=ax1)
    ax1.set_xticklabels(['Died', 'Survived'])
    st.pyplot(fig1)

    st.subheader("2. Age Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(filtered_df['Age'], bins=30, kde=True, ax=ax2)
    st.pyplot(fig2)

    st.subheader("3. Class-wise Survival")
    fig3, ax3 = plt.subplots()
    sns.countplot(data=filtered_df, x='Pclass', hue='Survived', ax=ax3)
    st.pyplot(fig3)

    st.subheader("4. Sex vs Survival")
    fig4, ax4 = plt.subplots()
    sns.countplot(data=filtered_df, x='Sex', hue='Survived', ax=ax4)
    st.pyplot(fig4)

    st.subheader("5. Fare Distribution")
    fig5, ax5 = plt.subplots()
    sns.histplot(filtered_df['Fare'], bins=20, ax=ax5)
    st.pyplot(fig5)

    st.subheader("6. Embarkation Distribution")
    fig6, ax6 = plt.subplots()
    sns.countplot(data=filtered_df, x='Embarked', ax=ax6)
    st.pyplot(fig6)

    st.subheader("7. Correlation Heatmap")
    fig7, ax7 = plt.subplots()
    sns.heatmap(filtered_df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax7)
    st.pyplot(fig7)

    # Download button
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download filtered data", csv, "filtered_data.csv", "text/csv")

    # Report / Conclusion
    with st.expander("📄 Project Report / Conclusion"):
        st.markdown("""
        **Observations:**
        - Survival rate appears higher for females than males.
        - First-class passengers had better survival chances.
        - Younger and older passengers showed different survival patterns.
        - Fare and class seem correlated.

        **Learnings:**
        - Explored data cleaning, filtering, and visualization techniques.
        - Practiced using seaborn and matplotlib with Streamlit.
        - Understood how interactive filters improve data exploration.
        """)

# ---------------------
# PART 2: IMAGE PROCESSING
# ---------------------
with tab2:
    st.header("Part 2: Image Processing with PIL")

    # Upload image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
    else:
        image = Image.open(r"E:\\Codes\\Advanced Python\\Dataset\\img\\morskie-oko-tatry.jpg")

    st.image(image, caption="Original Image", use_column_width=True)

    # Display image metadata
    st.markdown(f"**Format:** {image.format}, **Mode:** {image.mode}, **Size:** {image.size}")

    # Basic Operations
    st.subheader("Basic Transformations")
    cols = st.columns(2)
    cols[0].image(image.convert("L"), caption="Grayscale", use_column_width=True)
    cols[1].image(image.resize((150, 150)), caption="Resized (150x150)", use_column_width=True)

    cols2 = st.columns(2)
    cols2[0].image(image.rotate(90), caption="Rotated 90°", use_column_width=True)
    cols2[1].image(ImageOps.mirror(image), caption="Flipped Horizontally", use_column_width=True)

    # Filters
    st.subheader("Image Filters")
    filters = [
        ("Blur", image.filter(ImageFilter.BLUR)),
        ("Sharpen", image.filter(ImageFilter.SHARPEN)),
        ("Edge Enhance", image.filter(ImageFilter.EDGE_ENHANCE)),
        ("Contour", image.filter(ImageFilter.CONTOUR)),
        ("Emboss", image.filter(ImageFilter.EMBOSS)),
    ]

    for name, img in filters:
        st.image(img, caption=name, use_column_width=True)
