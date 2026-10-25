from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Insurance Charge Predictions", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")

DARK_BLUE, BRIGHT_BLUE = "#14558A", "#1680E5"
SMOKER_COLOURS = {"no": DARK_BLUE, "yes": BRIGHT_BLUE}
SEX_COLOURS = {"female": DARK_BLUE, "male": BRIGHT_BLUE}

st.markdown("""
<style>
.stApp{background:#102D70}
header[data-testid="stHeader"]{height:0!important;min-height:0!important;display:none!important}
.block-container{padding:0 .45rem .6rem .45rem!important;max-width:100%}
div[data-testid="stRadio"]{margin-top:0!important;margin-bottom:0!important;padding-top:0!important;padding-bottom:0!important}
div[data-testid="stVerticalBlock"]{gap:.25rem}
div[data-testid="stRadio"] label{font-weight:600}
.dashboard-header{background:#65A477;color:#252525;text-align:center;font-size:25px;font-weight:700;padding:9px 6px;border-bottom:4px solid #102D70;line-height:1.2}
.kpi-card{background:#A6BDD5;color:#252525;text-align:center;padding:9px 4px;border:2px solid #102D70;min-height:77px}
.kpi-value{font-size:27px;font-weight:500}.kpi-label{font-size:14px;color:#555}
.explanation-card{background:#A6BDD5;color:#102D70;border:1px solid #7898B8;border-radius:4px;padding:12px 12px;margin-top:16px}
.explanation-title{font-size:20px;font-weight:700;margin-bottom:10px;color:#102D70}
.explanation-body{font-size:17px;line-height:1.65;color:#102D70}
.filter-heading{display:block;color:#EAF2FF!important;font-weight:700;font-size:15px;line-height:1.35;margin:5;padding:0}
.accuracy-heading{display:block;color:#EAF2FF!important;font-weight:700;font-size:15px;line-height:1.35;margin:0;padding:0}
div[data-testid="stVerticalBlock"]{gap:.6rem}

div[data-testid="stMultiSelect"] [data-baseweb="select"] {
    min-height: 42px;
    margin-top: 5px;
}
.kpi-label{color:#334A62}
div[data-testid="stPlotlyChart"]{background:white;border-radius:3px;padding:2px}
div.stButton>button{background:#A6BDD5;color:#252525;border:1px solid #102D70;border-radius:2px;font-weight:600;min-height:54px;width:100%}
</style>""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "Output"
if not OUTPUT_DIR.exists():
    OUTPUT_DIR = BASE_DIR / "output"
CUSTOMER_PATH = OUTPUT_DIR / "cleaned_insurance.csv"
PREDICTION_PATH = OUTPUT_DIR / "insurance_predictions.csv"

@st.cache_data
def read_csv(path):
    return pd.read_csv(path)

for path in (CUSTOMER_PATH, PREDICTION_PATH):
    if not path.exists():
        st.error(f"Required dataset not found: {path}")
        st.stop()

customers = read_csv(str(CUSTOMER_PATH))
predictions = read_csv(str(PREDICTION_PATH))

def prepare(df):
    df = df.copy()
    for col in ["sex", "smoker", "region", "bmi_category"]:
        if col in df:
            df[col] = df[col].astype("string").str.strip().str.lower()
    if "age_category" not in df and "age" in df:
        df["age_category"] = pd.cut(
            df["age"], bins=[17,25,35,45,55,float("inf")],
            labels=["18-25 Young Adult","26-35 Adult","36-45 Middle-aged","46-55 Mature Adult","56+ Senior"],
            include_lowest=True)
    return df

customers, predictions = prepare(customers), prepare(predictions)

PAGES = ["Customer Characteristics", "Charge Distribution", "Machine-Learning Performance"]
if "active_page" not in st.session_state:
    st.session_state.active_page = PAGES[0]

# Horizontal navigation styled to remain visible above the dashboard header.
st.markdown("""
<style>
div[data-testid="stRadio"] {
    background-color: #A6BDD5;
    padding: 4px 10px;
    margin: 0 0 4px 0 !important;
    border-radius: 5px;
    width: 100%;
    box-sizing: border-box;
}
div[data-testid="stRadio"] > label {
    display: none;
}
div[data-testid="stRadio"] div[role="radiogroup"] {
    display: flex;
    flex-direction: row;
    justify-content: center;
    gap: 16px;
}
div[data-testid="stRadio"] div[role="radiogroup"] label {
    color: #102D70 !important;
    font-weight: 600;
    white-space: nowrap;
}
</style>
""", unsafe_allow_html=True)

page = st.radio(
    "Navigation",
    options=PAGES,
    horizontal=True,
    key="active_page",
    label_visibility="collapsed",
)

subtitle = {
    "Customer Characteristics":"Customer characteristics",
    "Charge Distribution":"Charge distribution",
    "Machine-Learning Performance":"Machine-learning performance"
}[page]
st.markdown(f'<div class="dashboard-header"><i>Insurance charge predictions</i><br>{subtitle}</div>', unsafe_allow_html=True)

FILTER_KEYS = ["filter_smoker","filter_bmi","filter_age","filter_region"]
for k in FILTER_KEYS:
    if k not in st.session_state:
        st.session_state[k] = []

def clear_filters():
    for k in FILTER_KEYS:
        st.session_state[k] = []

main_col, filter_col = st.columns([5.8,1.0], gap="small")
with filter_col:
    with st.container():
        st.markdown('<div class="filter-heading">Smoker</div>', unsafe_allow_html=True)
        st.multiselect(
            "Smoker",
            sorted(customers.smoker.dropna().unique().tolist()),
            key="filter_smoker",
            label_visibility="collapsed",
            placeholder="All smokers",
        )

    with st.container():
        st.markdown('<div class="filter-heading">BMI Category</div>', unsafe_allow_html=True)
        st.multiselect(
            "BMI Category",
            sorted(customers.bmi_category.dropna().unique().tolist()),
            key="filter_bmi",
            label_visibility="collapsed",
            placeholder="All BMI categories",
        )

    age_order = ["18-25 Young Adult","26-35 Adult","36-45 Middle-aged","46-55 Mature Adult","56+ Senior"]
    age_opts = [x for x in age_order if x in customers.age_category.astype(str).unique().tolist()]
    with st.container():
        st.markdown('<div class="filter-heading">Age Category</div>', unsafe_allow_html=True)
        st.multiselect(
            "Age Category",
            age_opts,
            key="filter_age",
            label_visibility="collapsed",
            placeholder="All age groups",
        )

    with st.container():
        st.markdown('<div class="filter-heading">Region</div>', unsafe_allow_html=True)
        st.multiselect(
            "Region",
            sorted(customers.region.dropna().unique().tolist()),
            key="filter_region",
            label_visibility="collapsed",
            placeholder="All regions",
        )

    # Explanatory panel remains beneath the filters.
    st.markdown("""
    <div class="explanation-card">
      <div class="explanation-title">About this page</div>
      <div class="explanation-body">
        {body}
      </div>
    </div>
    """.format(body={
        "Customer Characteristics": (
            "Explore how insurance charges vary with smoking status, age, region, BMI category and sex. "
            "Use the filters above to focus on a particular customer group. "
            "The smoker percentage is the share of filtered customers marked as smokers."
        ),
        "Charge Distribution": (
            "The histogram shows how charges are distributed, while box and violin plots compare spread, "
            "medians and unusually high values across groups. The scatter plot helps explore the relationship "
            "between BMI and charges. These are associations, not proof of causation."
        ),
        "Machine-Learning Performance": (
            "R² summarises how much variation the model explains; MAE is the average absolute prediction error; "
            "RMSE gives larger errors more weight. Error = predicted charges − actual charges: positive values "
            "mean overprediction and negative values mean underprediction. Errors within ±500 are labelled "
            "Accurate Prediction. Interpret the metrics in light of whether these predictions are from a held-out test set."
        )
    }[page]), unsafe_allow_html=True)

def apply_filters(df):
    out = df.copy()
    for key, col in [("filter_smoker","smoker"),("filter_bmi","bmi_category"),("filter_region","region")]:
        if st.session_state[key]:
            out = out[out[col].isin(st.session_state[key])]
    if st.session_state["filter_age"]:
        out = out[out.age_category.astype(str).isin(st.session_state["filter_age"])]
    return out

cust = apply_filters(customers)
pred = apply_filters(predictions)

def theme(fig, height=260, bottom=42):
    fig.update_layout(height=height, margin=dict(l=35,r=8,t=48,b=bottom), paper_bgcolor="white", plot_bgcolor="white", font=dict(color="#555"), legend=dict(orientation="h",yanchor="bottom",y=1.0,xanchor="left",x=0))
    fig.update_xaxes(gridcolor="#DDD"); fig.update_yaxes(gridcolor="#DDD")
    return fig

def kpis(items, key):
    cols = st.columns([1,1,1,1,1.65], gap="small")
    for col,(value,label) in zip(cols[:4],items):
        with col:
            st.markdown(f'<div class="kpi-card"><div class="kpi-value">{value}</div><div class="kpi-label">{label}</div></div>', unsafe_allow_html=True)
    with cols[4]:
        st.button("Clear all filters",on_click=clear_filters,key=key,use_container_width=True)

if page == "Customer Characteristics":
    with main_col:
        if cust.empty:
            st.warning("No customers match these filters. Clear the slicers."); st.stop()
        kpis([(f"{(cust.smoker=='yes').mean():.1%}","Smoker Percentage"),(f"{cust.bmi.mean():.2f}","Average BMI"),(f"{cust.charges.mean()/1000:.0f}K","Average Charges"),(f"{len(cust):,}","Total Customers")],"clear_customer")
        a,b,c=st.columns(3,gap="small")
        with a:
            d=cust.groupby("smoker",as_index=False).agg(avg=("charges","mean"))
            f=px.bar(d,x="smoker",y="avg",color="smoker",color_discrete_map=SMOKER_COLOURS,title="Average of charges by smoker",labels={"avg":"Average of charges"})
            f.update_layout(showlegend=False); f.update_yaxes(tickformat="~s"); st.plotly_chart(theme(f,255),use_container_width=True)
        with b:
            d=cust.groupby("region",as_index=False).agg(avg=("charges","mean")).sort_values("avg",ascending=False)
            f=px.bar(d,x="region",y="avg",color_discrete_sequence=[DARK_BLUE],title="Average of charges by region",labels={"avg":"Average of charges"})
            f.update_yaxes(tickformat="~s"); st.plotly_chart(theme(f,255),use_container_width=True)
        with c:
            f=px.scatter(cust,x="age",y="charges",color="smoker",color_discrete_map=SMOKER_COLOURS,title="Charges by age, smoker and BMI",hover_data=["bmi","region","sex"])
            f.update_yaxes(tickformat="~s"); st.plotly_chart(theme(f,255),use_container_width=True)
        a,b,c=st.columns(3,gap="small")
        with a:
            d=cust.groupby(["bmi_category","smoker"],as_index=False,observed=True).agg(avg=("charges","mean"))
            f=px.bar(d,x="bmi_category",y="avg",color="smoker",barmode="group",color_discrete_map=SMOKER_COLOURS,title="Average of charges by BMI category and smoker",labels={"avg":"Average of charges"})
            f.update_xaxes(tickangle=-30); f.update_yaxes(tickformat="~s"); st.plotly_chart(theme(f,275,60),use_container_width=True)
        with b:
            d=cust.groupby(["bmi_category","sex"],as_index=False,observed=True).agg(avg=("charges","mean"))
            f=px.bar(d,x="bmi_category",y="avg",color="sex",barmode="group",color_discrete_map=SEX_COLOURS,title="Average of charges by BMI category and sex",labels={"avg":"Average of charges"})
            f.update_xaxes(tickangle=-30); f.update_yaxes(tickformat="~s"); st.plotly_chart(theme(f,275,60),use_container_width=True)
        with c:
            f=px.histogram(cust,x="charges",y="charges",color="smoker",histfunc="sum",nbins=30,barmode="overlay",opacity=.8,color_discrete_map=SMOKER_COLOURS,title="Sum of charges by charges (bins) and smoker")
            f.update_xaxes(tickformat="~s"); f.update_yaxes(tickformat="~s"); st.plotly_chart(theme(f,275),use_container_width=True)

elif page == "Charge Distribution":
    with main_col:
        if cust.empty:
            st.warning("No customers match these filters. Clear the slicers."); st.stop()
        kpis([(f"{len(cust):,}","Total Customers"),(f"{cust.charges.mean()/1000:.2f}K","Average Charges"),(f"{cust.charges.median()/1000:.2f}K","Median Charges"),(f"{cust.charges.max()/1000:.2f}K","Maximum Charges")],"clear_distribution")
        a,b=st.columns(2,gap="small")
        with a:
            f=px.histogram(cust,x="charges",color="smoker",nbins=30,barmode="overlay",opacity=.75,color_discrete_map=SMOKER_COLOURS,title="Distribution of charges by smoking status")
            st.plotly_chart(theme(f,330),use_container_width=True)
        with b:
            f=px.box(cust,x="bmi_category",y="charges",color="smoker",color_discrete_map=SMOKER_COLOURS,title="Charge distribution by BMI category and smoker")
            f.update_xaxes(tickangle=-25); st.plotly_chart(theme(f,330,65),use_container_width=True)
        a,b=st.columns(2,gap="small")
        with a:
            f=px.violin(cust,x="region",y="charges",color="smoker",color_discrete_map=SMOKER_COLOURS,box=True,points=False,title="Charge distribution by region and smoker")
            st.plotly_chart(theme(f,330),use_container_width=True)
        with b:
            f=px.scatter(cust,x="bmi",y="charges",color="smoker",color_discrete_map=SMOKER_COLOURS,title="BMI versus charges by smoking status",hover_data=["age","region","sex"])
            st.plotly_chart(theme(f,330),use_container_width=True)

elif page == "Machine-Learning Performance":
    with main_col:
        if not {"charges","predicted_charges"}.issubset(pred.columns):
            st.error("Prediction file must contain charges and predicted_charges."); st.stop()
        ml=pred.copy()
        ml["charges"]=pd.to_numeric(ml["charges"],errors="coerce")
        ml["predicted_charges"]=pd.to_numeric(ml["predicted_charges"],errors="coerce")
        ml=ml.dropna(subset=["charges","predicted_charges"])
        if ml.empty:
            st.warning("No prediction rows match these filters."); st.stop()
        # Error sign follows the common convention: predicted minus actual.
        ml["error"]=ml["predicted_charges"]-ml["charges"]
        r2=r2_score(ml.charges,ml.predicted_charges) if len(ml)>1 else np.nan
        mae=mean_absolute_error(ml.charges,ml.predicted_charges)
        rmse=np.sqrt(mean_squared_error(ml.charges,ml.predicted_charges))
        kpis([(f"{len(ml):,}","Total Customers"),(f"{r2:.2f}" if np.isfinite(r2) else "N/A","R Squared"),(f"{mae/1000:.2f}K","MAE"),(f"{rmse/1000:.2f}K","RMSE")],"clear_ml")
        ml["accuracy_category"]=np.select([ml.error > 500,ml.error < -500],["Overprediction > 500","Underprediction < -500"],default="Accurate Prediction")
        accuracy=["Accurate Prediction","Overprediction > 500","Underprediction < -500"]
        st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="accuracy-heading">Prediction Accuracy Category</div>', unsafe_allow_html=True)
            st.markdown('<div style="height:5px"></div>', unsafe_allow_html=True)
            chosen = st.multiselect(
                "Prediction Accuracy Category",
                accuracy,
                default=[],
                placeholder="All prediction categories",
                label_visibility="collapsed",
                key="prediction_accuracy_category",
            )
        # An empty selection means all categories, matching the Power BI slicer convention.
        m=ml if not chosen else ml[ml.accuracy_category.isin(chosen)].copy()
        if m.empty:
            st.info("No rows match the selected accuracy categories.")
        else:
            a,b,c=st.columns(3,gap="small")
            with a:
                d=m.groupby(["bmi_category","smoker"],as_index=False,observed=True).agg(avg_error=("error","mean"))
                f=px.bar(d,x="bmi_category",y="avg_error",color="smoker",barmode="group",color_discrete_map=SMOKER_COLOURS,title="Average of Error by BMI category and smoker")
                f.add_hline(y=0,line_dash="dot",line_color="gray"); f.update_xaxes(tickangle=-30); st.plotly_chart(theme(f,280,60),use_container_width=True)
            with b:
                d=m.groupby(["bmi_category","sex"],as_index=False,observed=True).agg(avg_error=("error","mean"))
                f=px.bar(d,x="bmi_category",y="avg_error",color="sex",barmode="group",color_discrete_map=SEX_COLOURS,title="Average of Error by BMI category and sex")
                f.add_hline(y=0,line_dash="dot",line_color="gray"); f.update_xaxes(tickangle=-30); st.plotly_chart(theme(f,280,60),use_container_width=True)
            with c:
                d=m.groupby("smoker",as_index=False).agg(avg_error=("error","mean"))
                f=px.bar(d,x="smoker",y="avg_error",color="smoker",color_discrete_map=SMOKER_COLOURS,title="Average of Error by smoker")
                f.add_hline(y=0,line_dash="dot",line_color="gray"); f.update_layout(showlegend=False); st.plotly_chart(theme(f,280),use_container_width=True)
            a,b=st.columns([1.05,1.95],gap="small")
            with a:
                m["charges_bin"]=pd.cut(m.charges,bins=12,include_lowest=True)
                d=m.groupby(["charges_bin","smoker"],as_index=False,observed=True).agg(avg_error=("error","mean"))
                d["charges_bin"]=d.charges_bin.astype(str)
                f=px.bar(d,x="charges_bin",y="avg_error",color="smoker",barmode="group",color_discrete_map=SMOKER_COLOURS,title="Average of Error by charges (bins) and smoker")
                f.add_hline(y=0,line_dash="dot",line_color="gray"); f.update_xaxes(tickangle=-45); st.plotly_chart(theme(f,300,85),use_container_width=True)
            with b:
                f=px.scatter(m,x="predicted_charges",y="charges",color="smoker",color_discrete_map=SMOKER_COLOURS,title="Predicted charges versus actual charges",hover_data=["bmi_category","age","region"])
                lo=float(min(m.predicted_charges.min(),m.charges.min())); hi=float(max(m.predicted_charges.max(),m.charges.max()))
                f.add_shape(type="line",x0=lo,y0=lo,x1=hi,y1=hi,line=dict(color="gray",dash="dash"))
                f.update_xaxes(tickformat="~s"); f.update_yaxes(tickformat="~s"); st.plotly_chart(theme(f,300),use_container_width=True)


st.markdown('<div style="text-align:center;color:#D7E4F2;font-size:12px;padding-top:8px;">Insurance Charge Predictions | Assessment 3</div>',unsafe_allow_html=True)
