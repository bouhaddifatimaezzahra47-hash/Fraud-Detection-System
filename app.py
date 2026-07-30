import gradio as gr
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from google.colab import drive
import os
drive.mount('/content/drive')
df=pd.read_csv('/content/drive/MyDrive/FraudDetectionProject/creditcard.csv')
os.chdir("/content/drive/MyDrive/FraudDetectionProject")
model=joblib.load("grediant_boosting.pkl")
threshold=joblib.load("best_treshold3.pkl")
explainer_1=joblib.load("shap_explainer.pkl")
accuracy3=joblib.load("acuuracy3.pkl")
fig=joblib.load("fig.pkl")
precision3=joblib.load("precision3.pkl")
f13=joblib.load("f13.pkl")
fig2=joblib.load("fig2.pkl")
css = """
.gradio-container{
    background:#FCEFF3;
}

.metric-card{
    background:white;
    border-radius:18px;
    padding:20px;
    text-align:center;
    border:1px solid #f2c9d2;
    box-shadow:0px 4px 10px rgba(0,0,0,0.08);
    transition:0.3s;
}

.metric-card:hover{
    transform:translateY(-4px);
}

.metric-title{
    color:#666;
    font-size:16px;
    font-weight:600;
}

.metric-value{
    color:#E75480;
    font-size:32px;
    font-weight:bold;
    margin-top:10px;
}
"""
with gr.Blocks(css=css) as demo:
  with gr.Tab("Transactions Analysis"):

    gr.Markdown(f"""<h1 style="text-align:center;">Credit Card Fraud Detection Dashboard</h1>""")
    gr.Markdown(f"""<h3 style="text-align:center;">Explore transactions, analyze model predictions, and understand decisions using SHAP explanations.</h3>""")
    with gr.Row():
        total_card = gr.Markdown(f"""<div class="metric-card"><div class="metric-title">📊 Total Transactions</div><div class="metric-value">{len(df):,}</div></div>""")
        fraud_rate = df["Class"].mean()*100
        gr.Markdown(f"""<div class="metric-card"><div class="metric-title">🚨 Fraud Rate</div><div class="metric-value">{fraud_rate:.3f}%</div></div>""")
        recall=joblib.load("recall3.pkl")
        gr.Markdown(f"""<div class="metric-card"><div class="metric-title">🎯 Recall</div><div class="metric-value">{recall:.2f}%</div></div>""")
    transaction_index = gr.Slider(
        minimum=0,
        maximum=len(df)-1,
        step=1,
        value=0,
        label="Select Transaction Index"
    )

    analyze_btn = gr.Button("Analyze Transaction")
    with gr.Row():
      with gr.Column(scale=1):
        transaction_table = gr.Dataframe(headers=["Feature", "Value"],label="Transaction Details"
        )
      with gr.Column(scale=1):
        prediction = gr.Textbox(label="Prediction")
        shap_plot = gr.Plot(label="SHAP Explanation")
    def get_transaction(index):
      transaction = df.iloc[int(index)]
      table = pd.DataFrame({"Feature": transaction.drop("Class").index, "Value": transaction.drop("Class").values })
      x = transaction.drop("Class").to_frame().T
      probability = model.predict_proba(x)[0][1]
      shap_values = explainer_1.shap_values(x)
      shap_explanation = shap.Explanation(
        values=shap_values[0],
        base_values=explainer_1.expected_value,
        data=x.iloc[0],
        feature_names=x.columns)
      shap.plots.waterfall(
        shap_explanation,
        show=False)
      fig = plt.gcf()
      if probability >= threshold:
        prediction_text = "Fraud"
      else:
        prediction_text = "Normal"
      if transaction["Class"] == 1:
        actual = "Fraud"
      else:
        actual = "Normal"
      if prediction_text == actual:
        status = "✅ Correct"
      else:
        status = "❌ Incorrect"
      result = (
        f"Prediction : {prediction_text}\n\n"
         f"Fraud Probability : {probability:.2%}\n\n"
          f"Actual Class : {actual}\n\n"
        f"Status : {status}"
        )
      return table, result, fig
    analyze_btn.click(
    fn=get_transaction,
    inputs=transaction_index,
    outputs=[transaction_table, prediction, shap_plot]
)
  with gr.Tab("Model Performance"):
    gr.Markdown(f"""<h1 style='text-align:center;'>Model Performance</h1>""")
    gr.Markdown(f"""<h3 style='text-align:center;'>Evaluation metrics of the trained Gradient Boosting model</h3>""")
    with gr.Row():
       gr.Markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Accuracy</div>
            <div class="metric-value">{accuracy3:.2%}</div>
        </div>
        """)
       gr.Markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Precision</div>
            <div class="metric-value">{precision3:.2%}</div>
        </div>
        """)
       gr.Markdown(f"""
        <div class="metric-card">
            <div class="metric-title"> Recall</div>
            <div class="metric-value">{recall:.2%}</div>
        </div>
        """)
       gr.Markdown(f"""
        <div class="metric-card">
            <div class="metric-title"> F1 Score</div>
            <div class="metric-value">{f13:.2%}</div>
        </div>
        """)
       gr.Markdown(f"""
        <div class="metric-card">
          <div class="metric-title"> Decision Threshold</div>
          <div class="metric-value">{threshold:.4f}</div>
         </div>
    """)
    with gr.Row():
      with gr.Column(scale=1):
        gr.Markdown("## Confusion Matrix")
        gr.Plot(
        value=fig,
        label="Confusion Matrix"
        )
      with gr.Column(scale=1):
        gr.Markdown("## SHAP Feature Importance")
        gr.Plot(
            value=fig2,
            label="SHAP Summary"
        )




