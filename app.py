import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
)

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-size: 1rem;
        font-weight: bold;
        border: none;
        width: 100%;
    }
    .stButton>button:hover { background-color: #1d4ed8; }
    .result-box {
        background: linear-gradient(135deg, #1e3a8a, #2563eb);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin-top: 1.5rem;
    }
    .result-box h1 { font-size: 3rem; margin: 0; }
    .result-box p  { font-size: 1.1rem; margin-top: 0.5rem; opacity: 0.9; }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 6px rgba(0,0,0,0.08);
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load & Prepare Data (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_and_train():
   df = pd.read_csv("data/CarPrice_Assignment.csv")

    # Feature Engineering
    df['brand'] = df['CarName'].str.split().str[0].str.lower()
    brand_fix = {
        'alfa-romero': 'alfa-romeo', 'maxda': 'mazda',
        'toyouta': 'toyota', 'vw': 'volkswagen',
        'vokswagen': 'volkswagen', 'porcshce': 'porsche', 'Nissan': 'nissan'
    }
    df['brand'] = df['brand'].replace(brand_fix)
    df['doornumber']     = df['doornumber'].map({'two': 2, 'four': 4})
    df['cylindernumber'] = df['cylindernumber'].map(
        {'two':2,'three':3,'four':4,'five':5,'six':6,'eight':8,'twelve':12}
    )
    df['power_to_weight']   = df['horsepower'] / df['curbweight']
    df['engine_to_weight']  = df['enginesize']  / df['curbweight']
    df['car_volume']        = df['carlength'] * df['carwidth'] * df['carheight']
    df['avg_mpg']           = (df['citympg'] + df['highwaympg']) / 2
    df['bore_stroke_ratio'] = df['boreratio'] / df['stroke']
    luxury_brands = ['bmw', 'mercedes-benz', 'jaguar', 'porsche', 'buick', 'volvo']
    df['is_luxury']  = df['brand'].isin(luxury_brands).astype(int)
    df['log_price']  = np.log1p(df['price'])

    df.drop(columns=['car_ID', 'CarName'], inplace=True)

    cat_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    X = df.drop(columns=['price', 'log_price'])
    y = df['log_price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        min_samples_split=2, random_state=42
    )
    model.fit(X_train, y_train)

    return model, encoders, df, X.columns.tolist()

model, encoders, df, feature_cols = load_and_train()

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.title("🚗 Car Price Predictor")
st.markdown("Fill in the car details below and get an instant price estimate powered by Gradient Boosting ML.")

st.divider()

# ── Row 1: Brand & Body ──
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🏷️ Identity")
    brands = sorted([
        'alfa-romeo','audi','bmw','chevrolet','dodge','honda','isuzu',
        'jaguar','mazda','mercedes-benz','mitsubishi','nissan','plymouth',
        'porsche','renault','saab','subaru','toyota','volkswagen','volvo'
    ])
    brand = st.selectbox("Brand", brands)
    fueltype = st.selectbox("Fuel Type", ['gas', 'diesel'])
    aspiration = st.selectbox("Aspiration", ['std', 'turbo'])

with col2:
    st.subheader("🚙 Body")
    carbody = st.selectbox("Body Style", ['sedan', 'hatchback', 'wagon', 'hardtop', 'convertible'])
    drivewheel = st.selectbox("Drive Wheel", ['fwd', 'rwd', '4wd'])
    enginelocation = st.selectbox("Engine Location", ['front', 'rear'])
    doornumber = st.selectbox("Number of Doors", [2, 4])

with col3:
    st.subheader("📐 Dimensions")
    carlength  = st.slider("Length (inches)", 140, 210, 175)
    carwidth   = st.slider("Width (inches)",  60,  75, 67)
    carheight  = st.slider("Height (inches)", 45,  60, 54)
    curbweight = st.slider("Curb Weight (lbs)", 1500, 4500, 2500)

st.divider()

# ── Row 2: Engine ──
col4, col5, col6 = st.columns(3)

with col4:
    st.subheader("⚙️ Engine")
    enginetype  = st.selectbox("Engine Type", ['ohc', 'ohcv', 'ohcf', 'dohc', 'dohcv', 'l', 'rotor'])
    cylindernumber = st.selectbox("Cylinders", [2, 3, 4, 5, 6, 8, 12])
    enginesize  = st.slider("Engine Size (cc)", 60, 330, 130)
    fuelsystem  = st.selectbox("Fuel System", ['mpfi', '2bbl', 'mfi', '1bbl', 'spfi', '4bbl', 'idi', 'spdi'])

with col5:
    st.subheader("🔩 Performance")
    horsepower = st.slider("Horsepower", 48, 300, 102)
    boreratio  = st.slider("Bore Ratio", 2.5, 4.0, 3.2)
    stroke     = st.slider("Stroke", 2.0, 4.5, 3.2)
    compressionratio = st.slider("Compression Ratio", 7.0, 23.0, 9.0)

with col6:
    st.subheader("⛽ Fuel Economy")
    citympg    = st.slider("City MPG", 13, 50, 25)
    highwaympg = st.slider("Highway MPG", 16, 55, 30)
    peakrpm    = st.slider("Peak RPM", 4150, 6600, 5200)
    wheelbase  = st.slider("Wheelbase (inches)", 86, 121, 98)

st.divider()

# ─────────────────────────────────────────────
# Build Input Vector & Predict
# ─────────────────────────────────────────────
def encode(col_name, value):
    if col_name in encoders:
        try:
            return encoders[col_name].transform([str(value)])[0]
        except ValueError:
            return 0
    return value

if st.button("🔍 Predict Car Price"):
    # Engineered features
    power_to_weight   = horsepower / curbweight
    engine_to_weight  = enginesize  / curbweight
    car_volume        = carlength * carwidth * carheight
    avg_mpg           = (citympg + highwaympg) / 2
    bore_stroke_ratio = boreratio / stroke
    luxury_brands_set = ['bmw', 'mercedes-benz', 'jaguar', 'porsche', 'buick', 'volvo']
    is_luxury         = 1 if brand in luxury_brands_set else 0

    input_dict = {
        'symboling':       0,
        'fueltype':        encode('fueltype', fueltype),
        'aspiration':      encode('aspiration', aspiration),
        'doornumber':      doornumber,
        'carbody':         encode('carbody', carbody),
        'drivewheel':      encode('drivewheel', drivewheel),
        'enginelocation':  encode('enginelocation', enginelocation),
        'wheelbase':       wheelbase,
        'carlength':       carlength,
        'carwidth':        carwidth,
        'carheight':       carheight,
        'curbweight':      curbweight,
        'enginetype':      encode('enginetype', enginetype),
        'cylindernumber':  cylindernumber,
        'enginesize':      enginesize,
        'fuelsystem':      encode('fuelsystem', fuelsystem),
        'boreratio':       boreratio,
        'stroke':          stroke,
        'compressionratio':compressionratio,
        'horsepower':      horsepower,
        'peakrpm':         peakrpm,
        'citympg':         citympg,
        'highwaympg':      highwaympg,
        'brand':           encode('brand', brand),
        'power_to_weight': power_to_weight,
        'engine_to_weight':engine_to_weight,
        'car_volume':      car_volume,
        'avg_mpg':         avg_mpg,
        'bore_stroke_ratio':bore_stroke_ratio,
        'is_luxury':       is_luxury,
    }

    # Align to training feature order
    input_row = pd.DataFrame([[input_dict.get(f, 0) for f in feature_cols]], columns=feature_cols)

    log_pred   = model.predict(input_row)[0]
    pred_price = np.expm1(log_pred)

    # Display result
    st.markdown(f"""
    <div class="result-box">
        <p>Estimated Car Price</p>
        <h1>${pred_price:,.0f}</h1>
        <p>{'✨ Luxury Vehicle' if is_luxury else '🚗 Standard Vehicle'} &nbsp;|&nbsp; {brand.title()} &nbsp;|&nbsp; {horsepower} HP</p>
    </div>
    """, unsafe_allow_html=True)

    # Quick stats
    st.markdown("### 📊 Your Car at a Glance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Power-to-Weight", f"{power_to_weight:.3f}")
    m2.metric("Avg MPG", f"{avg_mpg:.1f}")
    m3.metric("Car Volume", f"{car_volume:,.0f} in³")
    m4.metric("Luxury?", "Yes ✅" if is_luxury else "No ❌")

    # Dataset comparison
    st.markdown("### 📈 How Does This Compare?")
    c1, c2 = st.columns(2)
    with c1:
        percentile = (df['price'] < pred_price).mean() * 100
        st.info(f"This price is higher than **{percentile:.0f}%** of cars in the dataset.")
    with c2:
        avg_price = df['price'].mean()
        diff = pred_price - avg_price
        st.info(f"Dataset average price is **${avg_price:,.0f}**. Your car is **{'$'+f'{abs(diff):,.0f}' } {'above' if diff > 0 else 'below'}** average.")

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.divider()
st.markdown(
    "<center style='color:gray;font-size:0.85rem;'>Built with ❤️ using scikit-learn & Streamlit &nbsp;|&nbsp; Model: Gradient Boosting &nbsp;|&nbsp; Dataset: UCI Car Price</center>",
    unsafe_allow_html=True
)
