import streamlit as st
import joblib
import plotly.graph_objects as go # Visualization အတွက်

# Page configuration (Website ခေါင်းစဉ်နဲ့ Icon သတ်မှတ်တာ)
st.set_page_config(page_title="HeartCare AI", page_icon="❤️")

# Load models
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

# --- SIDEBAR (Input အကွက်များကို ဘေးဘက်သို့ ရွှေ့ခြင်း) ---
st.sidebar.header("📋 လူနာအချက်အလက်များ")
st.sidebar.write("ကျေးဇူးပြု၍ အောက်ပါတို့ကို ဖြည့်စွက်ပါ")

age = st.sidebar.number_input("အသက် (Age)", 1, 100, 45)
chol = st.sidebar.number_input("ကိုလက်စထရော (Cholesterol)", 100, 500, 220)
hr = st.sidebar.number_input("အမြင့်ဆုံးနှလုံးခုန်နှုန်း (Max HR)", 50, 220, 150)

# --- MAIN PAGE (ပင်မမျက်နှာပြင်) ---
st.title("❤️ Heart Disease Prediction AI")
st.write("---") # မျဉ်းတားတားခြင်း

st.subheader("ခန့်မှန်းချက်ရလဒ် (Prediction Result)")

if st.sidebar.button("စစ်ဆေးမည် (Check Risk)"):
    features = [[age, chol, hr]]
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    prob = model.predict_proba(scaled_features)
    
    high_risk_percent = prob[0][1] * 100

    # UI Display
    col1, col2 = st.columns(2) # အကွက် ၂ ကွက် ခွဲလိုက်တာ

    with col1:
        if prediction[0] == 1:
            st.error("### High Risk ⚠️")
            st.write("နှလုံးရောဂါ ဖြစ်နိုင်ခြေ မြင့်မားနေပါသည်။ ဆရာဝန်နှင့် တိုင်ပင်ပါ။")
        else:
            st.success("### Low Risk ✅")
            st.write("နှလုံးရောဂါ ဖြစ်နိုင်ခြေ နည်းပါးပါသည်။ ကျန်းမာရေး ဆက်လက် ထိန်းသိမ်းပါ။")

    with col2:
        st.write(f"**ဖြစ်နိုင်ခြေ ရာခိုင်နှုန်း:** {high_risk_percent:.1f}%")
        # Progress bar လေးနဲ့ ပြတာ ပိုလှတယ်
        st.progress(int(high_risk_percent))

    st.write("---")
    st.info("မှတ်ချက်။ ။ ဤသည်မှာ AI မှ ခန့်မှန်းချက်သာ ဖြစ်ပါသည်။ ဆေးဘက်ဆိုင်ရာ ဆုံးဖြတ်ချက်များအတွက် ကျွမ်းကျင်ဆရာဝန်နှင့် ပြသပါ။")



    # ------------------------------ plotly chart
    # Gauge Chart ဆွဲမယ်
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = high_risk_percent,
        title = {'text': "Risk Probability %"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"}
            ],
        }
    ))
    st.plotly_chart(fig)


else:
    st.info("ဘေးဘက်ရှိ Sidebar တွင် အချက်အလက်များဖြည့်ပြီး 'Check Risk' ကို နှိပ်ပါ။")