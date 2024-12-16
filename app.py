import streamlit as st
import pandas as pd
from pycaret.classification import *
from streamlit_option_menu import option_menu
from pycaret.datasets import get_data

st.set_page_config(layout="wide")

model = load_model('clasf_smart_electrical')

def predict(model, input_df):
    predictions_df = predict_model(model, data=input_df)
    return predictions_df

def prediction_page():
    st.title('Electrical Grid Stability Prediction')
    st.markdown('This app predicts the stability of Electrical Grid based on input features.')

    input_option = st.radio("**Choose input method:**", ('Manual Input', 'CSV Upload'))

    if input_option == 'Manual Input':
        # Input form for manual input
        st.header('Input Features')

        col1, col2, col3 = st.columns(3)

        with col1:
            tau1 = st.number_input('Reaction Time (tau1 (s))', min_value=0.5, max_value=10.0, format="%.6f")
            tau2 = st.number_input('Reaction Time (tau2 (s))', min_value=0.5, max_value=10.0, format="%.6f")
            tau3 = st.number_input('Reaction Time (tau3 (s))', min_value=0.5, max_value=10.0, format="%.6f")
            tau4 = st.number_input('Reaction Time (tau4 (s))', min_value=0.5, max_value=10.0, format="%.6f")

        with col2:
            p1 = st.number_input('Power (p1 (s^-2))', value=0.0, format="%.6f")
            p2 = st.number_input('Power (p2 (s^-2))', value=0.0, format="%.6f")
            p3 = st.number_input('Power (p3 (s^-2))', value=0.0, format="%.6f")
            p4 = st.number_input('Power (p4 (s^-2))', value=0.0, format="%.6f")
            
        with col3:
            g1 = st.number_input('Price Elasticity Coefficient (g1 (s^-1))', min_value=0.05, max_value=1.00, format="%.6f")
            g2 = st.number_input('Price Elasticity Coefficient (g2 (s^-1))', min_value=0.05, max_value=1.00, format="%.6f")
            g3 = st.number_input('Price Elasticity Coefficient (g3 (s^-1))', min_value=0.05, max_value=1.00, format="%.6f")
            g4 = st.number_input('Price Elasticity Coefficient (g4 (s^-1))', min_value=0.05, max_value=1.00, format="%.6f")

        # Transform inputs to match the model's requirements
        input_dict = {
            'Reaction Time (tau1)': tau1,
            'Reaction Time (tau2)': tau2,
            'Reaction Time (tau3)': tau3,
            'Reaction Time (tau4)': tau4,
            'Power (p1)': p1,
            'Power (p2)': p2,
            'Power (p3)': p3,
            'Power (p4)': p4,
            'Price Elasticity Coefficient (g1)': g1,
            'Price Elasticity Coefficient (g2)': g2,
            'Price Elasticity Coefficient (g3)': g3,
            'Price Elasticity Coefficient (g4)': g4
        }
        input_df = pd.DataFrame([input_dict])

        if st.button('Predict'):
            prediction = predict(model=model, input_df=input_df)
            st.write(prediction)
            if prediction['prediction_label'][0] == 'unstable':
                st.error(f"Predicted Stability: **Unstable** with a prediction accuracy score of {prediction['prediction_score'][0]:.4f}")
            else:
                st.success(f"Predicted Stability: **Stable** with a prediction accuracy score of {prediction['prediction_score'][0]:.4f}")

    else:  # CSV Upload
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            try:
                input_df = pd.read_csv(uploaded_file)
                if st.button('Predict'):
                    prediction = predict(model=model, input_df=input_df)
                    st.write(prediction)
            except Exception as e:
                st.error("An error occurred while processing the file. Make sure it's a CSV file with valid Electrical Grid Feature Data!")

def home_page():
    st.title("Welcome to MyApp!")
    st.write("This Streamlit application serves as a tool for predicting the stability of electrical grids based on various input features.")
    
    st.header("**💾 About Dataset**")
    col1, col2, col3 = st.columns(3)
    with col2:
        st.image('images/electrical_grid.png', width=500,  caption='Electrical Grid')
    st.markdown("""
            <div style='text-align: justify;'>
                The stability of electrical grids refers to the ability of the power system or its components to maintain synchronization and balance within the system during disturbances or load changes. 
                In power systems, stability means that voltage and frequency remain constant, and generating machines are able to operate in stable synchronization despite disturbances. 
                The dataset used in this modeling consists of 10,000 entries, with each entry having 13 relevant attributes or features. 
            </div>
            """, unsafe_allow_html=True)

    data = {
        "Feature": ["Reaction Time (tau1, tau2, tau3, tau4)",
                    "Power (p1, p2, p3, p4)",
                    "Price Elasticity Coefficient (g1, g2, g3, g4)",
                    "Stability"],
        "Type": ["Primary, independent, predictive",
                "Primary, independent, predictive",
                "Primary, independent, predictive",
                "Secondary and dependent (Categorical)"],
        "Significance": ["Representing producer's reaction time and three consumer's reaction time (range: from 0.5 to 10)",
                        "Representing power generated (positive values) and consumed (negative values) by producer and consumers, respectively",
                        "Price elasticity coefficient of producer and consumer (range: from 0.05 to 1.00)",
                        "Categorical feature, binary-valued labeled stable/unstable"]
    }

    st.table(data)

    st.header("**📂 Data Preview**")
    data = get_data('electrical_grid')
    data = data.rename(columns={
    'tau1': 'Reaction Time (tau1)',
    'tau2': 'Reaction Time (tau2)',
    'tau3': 'Reaction Time (tau3)',
    'tau4': 'Reaction Time (tau4)',
    'p1': 'Power (p1)',
    'p2': 'Power (p2)',
    'p3': 'Power (p3)',
    'p4': 'Power (p4)',
    'g1': 'Price Elasticity Coefficient (g1)',
    'g2': 'Price Elasticity Coefficient (g2)',
    'g3': 'Price Elasticity Coefficient (g3)',
    'g4': 'Price Elasticity Coefficient (g4)',
    'stabf': 'Stability'
    })

    st.dataframe(data.head())

    st.header("**⚙️ Modeling**")
    st.markdown("""
                <div style="text-align: center;">
                    <div style="text-align: justify; display: inline-block;">
                        This modeling uses <strong>CatBoost Classification</strong> with the target feature being <strong>Stability</strong>.
                        <strong>CatBoost stands for “Categorical Boosting,”</strong> and it is a machine learning algorithm that falls under the gradient boosting framework. As its name suggests,
                        CatBoost has two main features, it works with categorical data and it uses gradient-boosting algorithms for inferences. Gradient boosting is an ensemble technique 
                        that combines the predictions from multiple weak learners (typically decision trees) to create a strong predictive model. What sets CatBoost apart is its ability to 
                        handle categorical features efficiently, without the need for preprocessing, and its strong performance out of the box.
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.header("**🔮 Prediction Results**")
    st.markdown("""
                After selecting the input features, the application will predict whether the electrical grid is stable or unstable based on the provided data. 
                The prediction will be accompanied by the probability score indicating the confidence level of the prediction.
                """)

    st.header("**⚓ Stability Definitions**")
    st.markdown("""
                - **Stable Electrical Grid**: A stable electrical grid is capable of providing consistent power supply in terms of voltage and frequency.
                - **Unstable Electrical Grid**: An unstable electrical grid is unable to provide consistent power supply in terms of voltage and frequency.
                """)

def about_me():
    st.title("About Me")
    st.markdown(
        """
        **Welcome to the "About Me" page!** Let me introduce myself briefly.
        """
    )
    col1, col2, col3 = st.columns(3)
    with col2:
        st.image("images/fprofil.png", width=300)

    st.write("## Short Bio")
    st.write(
        """
        <div style='text-align: justify;'>
        Hello, I'm Johan Winarwan Nawawi, a student of Surabaya State Electronics Polytechnic, majoring in Applied Data Science.
        I am very interested in the field of data analysis and am proficient in using Python for data manipulation and analysis.
        My goal is to utilize my skills and knowledge to excel in the role of data analyst and contribute meaningfully to projects in this field.
        </div>
        """, 
        unsafe_allow_html=True
    )

    st.write("## Contact")
    st.write(
        """
        If you'd like to get in touch with me, feel free to send me an [Email](mailto:winarwanjohan@gmail.com) or [WhatsApp](https://wa.me/628817172827).
        """
    )

    st.write("## Social Media")
    st.write(
        """
        Follow me on [![Instagram](https://img.shields.io/badge/Instagram-Follow-lightgrey?style=social&logo=instagram)](https://www.instagram.com/_johannawawi) 
        and [![GitHub](https://img.shields.io/badge/GitHub-Follow-lightgrey?style=social&logo=github)](https://github.com/johannawawi) 
        for updates on my latest projects and discoveries
        """
    )

def main():
    st.sidebar.title('Navigation')
    st.sidebar.image("images/header.png", width=200)
    with st.sidebar:
        selected = option_menu(
                menu_title=None,
                options= ["Home", "Prediction", "About Me"],
                icons=['house', 'book', 'person'],
                menu_icon='cast',
                orientation='vertical',
                styles={
                        "container": {"padding": "0!important", "background-color": "#fafafa"},
                        "icon": {"color": "#ffd700", "font-size": "15px"},
                        "nav-link": {
                            "font-size": "15px",
                            "text-align": "left",
                            "margin": "0px",
                            "--hover-color": "#eee",
                        },
                        "nav-link-selected": {"background-color": "#0066b2"},
                    },
                )

    if selected == "Home":
        home_page()
    elif selected == "Prediction":
        prediction_page()
    else:
        about_me()
    st.sidebar.warning('Use this application to predict the stability of electrical grids based on the available features!')
    st.sidebar.title(' ')
    st.sidebar.title(' ')
    st.sidebar.title(' ')
    st.sidebar.title(' ')
    st.sidebar.title(' ')
    st.sidebar.info('Sains Data Terapan, :copyright:2024')
    st.sidebar.success('Created by: Johan Winarwan Nawawi')

if __name__ == '__main__':
    main()