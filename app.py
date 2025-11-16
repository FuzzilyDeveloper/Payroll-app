import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Sri Lanka Payroll Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Title & Header
# -----------------------------
st.title("🇱🇰 Sri Lanka Payroll Analytics Dashboard")
st.markdown("### November 2025 • 150 Employees • Interactive Visualization & ML Salary Prediction")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("sri_lanka_payroll_nov_2025.csv")
        df['Join_Date'] = pd.to_datetime(df['Join_Date'])
        return df
    except FileNotFoundError:
        st.error("⚠️ File `sri_lanka_payroll_nov_2025.csv` not found! Please upload it.")
        st.stop()

df = load_data()

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("🔍 Filters")

selected_depts = st.sidebar.multiselect(
    "Department",
    options=sorted(df['Department'].unique()),
    default=sorted(df['Department'].unique())
)

selected_positions = st.sidebar.multiselect(
    "Position",
    options=sorted(df[df['Department'].isin(selected_depts)]['Position'].unique()),
    default=[]
)

gross_range = st.sidebar.slider(
    "Gross Salary Range (LKR)",
    min_value=int(df['Gross_Salary'].min()),
    max_value=int(df['Gross_Salary'].max()),
    value=(int(df['Gross_Salary'].quantile(0.1)), int(df['Gross_Salary'].quantile(0.9)))
)

# Apply filters
filtered_df = df[
    df['Department'].isin(selected_depts) &
    (df['Position'].isin(selected_positions) if selected_positions else True) &
    df['Gross_Salary'].between(gross_range[0], gross_range[1])
]

st.sidebar.metric("Employees Displayed", len(filtered_df))
st.sidebar.caption(f"Total Records: {len(df)}")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔍 Explore Data", "🤖 ML Prediction", "📄 Raw Data"])

# =============================
# Tab 1: Dashboard
# =============================
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Gross Payroll", f"Rs. {filtered_df['Gross_Salary'].sum():,.0f}")
    col2.metric("Total Net Paid", f"Rs. {filtered_df['Net_Salary'].sum():,.0f}")
    col3.metric("Total PAYE Tax", f"Rs. {filtered_df['PAYE_Tax'].sum():,.0f}")
    col4.metric("Average Net Salary", f"Rs. {filtered_df['Net_Salary'].mean():,.0f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        dept_avg = filtered_df.groupby('Department')['Net_Salary'].mean().sort_values(ascending=False)
        fig_bar = px.bar(dept_avg, title="Average Net Salary by Department", color=dept_avg.index, text_auto=True)
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

        fig_box = px.box(filtered_df, x='Department', y='Gross_Salary', color='Department',
                         title="Gross Salary Distribution by Department")
        st.plotly_chart(fig_box, use_container_width=True)

    with col2:
        fig_hist = px.histogram(filtered_df, x='Net_Salary', nbins=30, marginal="box",
                                title="Net Salary Distribution", color_discrete_sequence=["#e74c3c"])
        st.plotly_chart(fig_hist, use_container_width=True)

        dept_pie = filtered_df['Department'].value_counts()
        fig_pie = px.pie(values=dept_pie.values, names=dept_pie.index, title="Employee Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)

    # Tax vs Salary
    fig_scatter = px.scatter(filtered_df, x='Gross_Salary', y='PAYE_Tax',
                             color='Department', size='EPF_Employee_8%',
                             hover_data=['Full_Name', 'Position'],
                             title="PAYE Tax vs Gross Salary (Bubble = EPF 8%)")
    st.plotly_chart(fig_scatter, use_container_width=True)

# =============================
# Tab 2: Data Explorer
# =============================
with tab2:
    st.dataframe(
        filtered_df.style.format({
            'Basic_Salary': 'Rs. {:,.0f}',
            'Gross_Salary': 'Rs. {:,.0f}',
            'Net_Salary': 'Rs. {:,.0f}',
            'PAYE_Tax': 'Rs. {:,.0f}',
            'EPF_Employee_8%': 'Rs. {:,.0f}'
        }),
        use_container_width=True
    )

    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_payroll_nov_2025.csv",
        mime="text/csv"
    )

# =============================
# Tab 3: ML Salary Prediction
# =============================
with tab3:
    st.markdown("### 🤖 Predict Net Salary Using Machine Learning")
    st.info("Trained on current dataset • Random Forest Regressor")

    # Prepare model
    model_df = df.copy()
    model_df = pd.get_dummies(model_df, columns=['Department', 'Position'], drop_first=False)

    feature_cols = [c for c in model_df.columns if c not in [
        'Employee_ID', 'Full_Name', 'Join_Date', 'Payment_Month',
        'Net_Salary', 'Gross_Salary', 'Taxable_Income'
    ]]

    X = model_df[feature_cols].select_dtypes(include=[np.number])
    y = model_df['Net_Salary']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    col1, col2 = st.columns(2)
    col1.metric("Model R² Score", f"{r2_score(y_test, y_pred):.3f}")
    col2.metric("Prediction Error (MAE)", f"± Rs. {mean_absolute_error(y_test, y_pred):,.0f}")

    st.success("Model trained successfully!")

    st.markdown("#### Predict Net Salary for a New Employee")

    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            dept = st.selectbox("Department", options=df['Department'].unique())
            basic = st.number_input("Basic Salary (LKR)", 50000, 600000, 150000, 10000)
            attendance = st.number_input("Attendance Allowance", 0, 30000, 15000, 5000)
            transport = st.number_input("Transport Allowance", 0, 20000, 10000, 5000)
        with c2:
            pos = st.selectbox("Position", options=df[df['Department']==dept]['Position'].unique())
            meal = st.number_input("Meal Allowance", 0, 15000, 5000, 1000)
            bonus = st.number_input("Performance Bonus", 0, 200000, 0, 10000)
            no_pay = st.slider("No-Pay Days", 0.0, 5.0, 0.0, 0.5)

        submitted = st.form_submit_button("🔮 Predict Net Salary")

        if submitted:
            gross = basic + attendance + transport + meal + bonus
            epf = gross * 0.08
            taxable = gross - epf

            # Simple PAYE calculation
            brackets = [(0,100000,0), (100001,191667,0.06), (191668,283333,0.12),
                        (283334,375000,0.18), (375001,466667,0.24), (466668,1e9,0.30)]
            paye = 0
            prev = 0
            for low, high, rate in brackets:
                if taxable > prev:
                    paye += min(taxable, high) - prev * rate
                prev = high

            deductions = epf + paye + (gross/30)*no_pay + 225  # approx other deductions
            rule_based_net = gross - deductions

            # ML Prediction
            input_data = pd.DataFrame([{col: 0 for col in X.columns}])
            input_data['Basic_Salary'] = basic
            input_data['Attendance_Allowance'] = attendance
            input_data['Transport_Allowance'] = transport
            input_data['Meal_Allowance'] = meal
            input_data['Performance_Bonus'] = bonus
            input_data['EPF_Employee_8%'] = epf
            input_data['PAYE_Tax'] = paye
            input_data['No_Pay_Deduction'] = (gross/30)*no_pay
            input_data[f'Department_{dept}'] = 1
            input_data[f'Position_{pos}'] = 1

            input_data = input_data.reindex(columns=X.columns, fill_value=0)
            ml_prediction = model.predict(input_data)[0]

            st.markdown(f"### Predicted Net Salary: **Rs. {ml_prediction:,.0f}**")
            st.info(f"Rule-based calculation: Rs. {rule_based_net:,.0f}")

# =============================
# Tab 4: Raw Data
# =============================
with tab4:
    st.dataframe(df.style.format({
        'Basic_Salary': 'Rs. {:,.0f}',
        'Gross_Salary': 'Rs. {:,.0f}',
        'Net_Salary': 'Rs. {:,.0f}',
        'PAYE_Tax': 'Rs. {:,.0f}'
    }), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("💼 Sri Lanka Payroll Dashboard • November 2025 • Built with Streamlit")
