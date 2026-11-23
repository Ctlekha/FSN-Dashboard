from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
ROOT=Path(__file__).parent
st.set_page_config(page_title="FSN Analytics",layout="wide")
st.markdown("""<style>.stApp{background:#F1F5F9}.block-container{max-width:1200px;padding-top:2rem} [data-testid="stSidebar"]{display:none} [data-testid="stMetric"]{background:white;border:1px solid #dce5f0;border-radius:18px;padding:22px;box-shadow:0 4px 16px #14233e09} [data-testid="stMetricValue"]{color:#1e3a8a} [data-testid="stPlotlyChart"]{background:white;border-radius:18px;padding:12px;border:1px solid #dce5f0} .hero{background:linear-gradient(110deg,#0F172A,#1E3A5F);border-radius:22px;padding:30px;margin:12px 0 24px;color:white}.hero h1{color:white;margin:0}.hero p{color:#dbeafe;margin:8px 0} [data-testid="stRadio"]{background:white;border-radius:14px;padding:12px;border:1px solid #dce5f0} h1,h2,h3{color:#14233e}</style>""",unsafe_allow_html=True)
st.markdown("### FSN / Analytics")
page=st.radio("Navigation",["Overview","Data Quality Assessment","Model Performance","Final Comparison"],horizontal=True,label_visibility="collapsed")
a,b=st.columns([4,1]); a.caption("RESEARCH INSIGHTS  /  UNIVERSITI MALAYSIA KELANTAN"); y=b.selectbox("Dataset",["2022","2023"],label_visibility="collapsed") if page!="Final Comparison" else "2022"
@st.cache_data
def read(y,f): return pd.read_csv(ROOT/"data"/y/(f+".csv"))
def bar(df,x,cols,label="Score (%)"):
    a=df.melt(id_vars=x,value_vars=cols,var_name="Series",value_name=label)
    a[label]=pd.to_numeric(a[label],errors="coerce")
    fig=px.bar(a.dropna(subset=[label]),x=x,y=label,color="Series",barmode="group",color_discrete_sequence=["#1E3A8A","#93C5FD"],text_auto=".2f")
    if label=="Score (%)": fig.update_yaxes(range=[0,110])
    fig.update_layout(template="plotly_white",height=350,legend_title_text="",margin=dict(t=15,b=10,l=10,r=10))
    st.plotly_chart(fig,use_container_width=True)
metrics=["Accuracy","F1-Macro","Balanced Accuracy","Minority Recall","ROC-AUC"]
st.markdown(f'<div class="hero"><small>FOOD SECURITY &amp; NUTRITION</small><h1>{page}</h1><p>Data Quality Assessment Framework | {y if page!="Final Comparison" else "2022 — 2023"}</p></div>',unsafe_allow_html=True)
final=read(y,"59_37").set_index("Metric")
config=read(y,"59_35").set_index("Parameter").Value
if page=="Overview":
    info=read(y,"59_1").set_index("Item").Value
    cs=st.columns(4)
    cs[0].metric("Total records",f"{int(info['Original Records']):,}")
    for c,m in zip(cs[1:],metrics[:3]): c.metric(m,f"{final.loc[m,'Testing (%)']:.2f}%")
    st.subheader("Target distribution")
    raw=read(y,"5_11").copy(); raw["Category"]=raw.iloc[:,0].str.lower().str.replace(" ","",regex=False).map({"error/tiadadata":"Error / No Data","malpemakanan":"Malnutrition","normal":"Normal","outofrange":"Out of Range"})
    assert raw.Category.notna().all(), "Unexpected target label"
    eligible=raw[raw.Category.isin(["Normal","Malnutrition"])].copy()
    colors={"Error / No Data":"#F87171","Malnutrition":"#3B82F6","Normal":"#22C55E","Out of Range":"#CBD5E1"}
    def donut(d,title):
       fig = px.pie(d, names="Category", values="Count", color="Category", color_discrete_map=colors, hole=.68)
       fig.update_traces(sort=False)
        fig.update_traces(textinfo="percent",hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>")
        fig.add_annotation(text=f"<b>{int(d.Count.sum()):,}</b><br>records",x=.5,y=.5,showarrow=False,font=dict(size=17,color="#0F172A"))
        fig.update_layout(title=dict(text=title,x=.5,font=dict(size=16)),template="plotly_white",height=360,legend=dict(orientation="h",y=-.08),margin=dict(t=55,b=60,l=15,r=15))
        st.plotly_chart(fig,use_container_width=True)
    left,right=st.columns(2)
    with left: donut(raw,"Original target distribution")
    with right: donut(eligible,"Eligible binary records")
    st.info(f"{eligible.Count.sum():,} of {raw.Count.sum():,} records ({eligible.Count.sum()/raw.Count.sum():.2%}) are eligible for binary classification. Error / No Data and Out of Range are original labels, not model prediction errors. They are excluded from binary classification; they are not automatically proven invalid.")
    st.subheader("Final testing performance")
    d=final.loc[metrics,["Testing (%)"]].reset_index(); fig=px.bar(d,x="Testing (%)",y="Metric",orientation="h",text_auto=".2f",color_discrete_sequence=["#3B82F6"]); fig.update_layout(template="plotly_white",height=310,yaxis=dict(autorange="reversed"),xaxis=dict(range=[0,105]),margin=dict(t=10,b=10,l=10,r=20)); st.plotly_chart(fig,use_container_width=True)
    st.subheader("Selected framework")
    steps=config["Final Model Sequence"].split(" -> ")
    for i,(col,step) in enumerate(zip(st.columns(len(steps)),steps)):
        col.markdown(f'<div style="background:white;border:1px solid #dce5f0;border-radius:16px;padding:20px"><small>STAGE {i+1:02d}</small><br><strong>{step}</strong></div>',unsafe_allow_html=True)
elif page=="Data Quality Assessment":
    d=read(y,"59_5").set_index("Metric")
    c=st.columns(3)
    for col,m in zip(c,["Missing Cells","Invalid Numeric Values","Duplicate Records"]): col.metric(m,f"{int(d.loc[m,'After']):,}",f"{int(d.loc[m,'Change']):+,} vs before",delta_color="off")
    st.subheader("Before vs after assessment")
    bar(read(y,"59_3"),"Dimension",["Before (%)","After (%)"])
    st.caption("Accuracy and timeliness: not assessable. Completeness decreased as hidden missing and invalid values were exposed; validity is not measurement accuracy.")
elif page=="Model Performance":
    model=st.selectbox("Algorithm",["DBSCAN","Random Forest","K-Means","Neural Network"])
    st.caption("Stage experiments (validation), not final testing.")
    if model=="DBSCAN":
        bar(read(y,"59_7"),"Strategy",["Accuracy (%)","F1-Macro (%)"])
        st.write("Final strategy: "+config["DBSCAN Filtering"])
    elif model=="Random Forest":
        c=st.columns(2); c[0].metric("Encoded features",config["Original Encoded Features"]); c[1].metric("Selected features",config["Selected RF Features"])
        st.write("Random Forest retained for feature selection in the final framework.")
    elif model=="K-Means":
        d=read(y,"59_17"); bar(d[d.Metric.isin(["Accuracy (%)","F1-Macro (%)","Balanced Accuracy (%)"])],"Metric",["Without Clustering","With Clustering"])
        st.write("Final configuration: without clustering.")
    else:
        d=read(y,"59_19"); bar(d[d.Metric.isin(metrics[:4])],"Metric",["Baseline (%)","Optimized (%)"])
        st.write("Neural Network is the final classifier.")
else:
    d=pd.DataFrame({year:read(year,"59_37").set_index("Metric")["Testing (%)"] for year in ["2022","2023"]}).loc[metrics].reset_index()
    st.subheader("Final testing performance")
    bar(d,"Metric",["2022","2023"])
    st.dataframe(d.style.format({"2022":"{:.3f}","2023":"{:.3f}"}),hide_index=True,use_container_width=True)
    st.caption("Separate within-year testing; not cross-year transfer. Neither dataset reached the 80% accuracy target.")
    st.download_button("Download results",d.to_csv(index=False),"final_comparison.csv","text/csv")
