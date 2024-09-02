import streamlit as st
import numpy as np
import requests
import joblib
import matplotlib.pyplot as plt


api_url = "http://127.0.0.1:8000/predict_cluster/"

# Set the page configuration

st.set_page_config(page_title=" ", layout="centered")

st.markdown(
    """
    <div style="text-align: center; ">
        <h1 style="color: #2e3b4e;">يوم مفتوح في طويق</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.image('dish_party.jpg')


# Apply custom styles for right-to-left alignment and better formatting
st.markdown(
    """
    <style>
    .rtl-text {
        text-align: right;
        direction: rtl;
        font-size: 18px;
        line-height: 1.6;
    }
    .title {
        font-size: 24px;
        font-weight: bold;
        color: #2e3b4e;
        text-align: center;
        direction: rtl;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display the title

# Display the content
st.markdown(
    """
    <div class="rtl-text">
    في يوم طبيعي زي كل يوم ضغط من المهندسة إسراء وبعض المساعدين (ريفال) ,فجأة وبدون مقدمات المهندسة إسراء مع نهاية الدرس الطويل جداً تقول لنا انه الاسبوع الجاي فيه dish party .
    <br>كل الكلاس:  dish party!!!?!!? 
    <br>صوت من بعيييييييد: يعني يوم مفتوح؟؟؟؟؟؟ 
    <br><br>بعد ما شرحت لنا المهندسة إسراء وطلع فعلاً هو اليوم المفتوح بس لابس كرفتّه.
    <br><br> إختارت القصمان اللي بالكلاس فراس وهتون وجدو سلمان عشان يكون مشرف علينا, طبعاً زي ما تعرفون موديله 1990
    <br><br>وقالت لنا انه إختيار المطاعم عليكم ومن هنا بدأت معاناة الإختيار وبعض الإقتراحات اللي مالها داعي و بحكم إن أغلبية الكلاس من برا الرياض قررنا نخرج عالم البيانات الغلبان اللي بداخلنا وندور داتا تساعدنا بالإختيار.
    <br>ووقع الإختيار على موقع Welcome Saudi
    <br><br>طبعاً نحن أشخاص ديموقراطيين أخذنا تصويت الكلاس وبناءً عليه حددنا 3 مطابخ مختلفة: 
    <br>ايطالي , اسيوي , عربي.
    <br><br>بعد ما خلصنا من اختيار الغداء جلسنا نفكر بموضوع الحلا من وين ناخذه 
    <br><br>وبعد ما انتهينا من إختيار المطاعم بدأت هتون بجمع القطة.
    <br>وبعد ما اكتملت القطة قررت هتون الله يهديها تعطي الفلوس لفراس وزي ما كنا متوقعيين, فراس سحب وإختفى وجلسنا نفكر ب اكثر من سيناريو أولها اننا انزرفنا وبنرجع للبيت جوعانين 
    <br><br>وجات المهندسة الاسطورية إسراء وانقذت الموقف لانها اغنى وحده فينا ودفعت الفلوس والى الان القصة وراء إختفاء فراس مجهولة!!!!!!
    </div>



    <br> طيب الحين لو جاء يوم مفتوح ثاني؟؟؟ كيف نعرف أحسن المطاعم؟ احنا بنزبطكم اليوم كل الي عليك انك تختار نوعية الاكل الي تبغاه وبنطلع لك الافضل تقييما ;)  
    """,
    unsafe_allow_html=True
)


cuisine = st.selectbox(
    "اختر نوع المطبخ:",
    ("Middle Eastern", "Italian", "Asian", "Cafes")
)

# Display corresponding image based on the selection
if cuisine == "Middle Eastern":
    st.image("ME.PNG", caption="Middle Eastern Cuisine", use_column_width=True)
elif cuisine == "Italian":
    st.image("Italian.PNG", caption="Italian Cuisine", use_column_width=True)
elif cuisine == "Asian":
    st.image("Asian.PNG", caption="Asian Cuisine", use_column_width=True)
elif cuisine == "Cafes":
    st.image("Cafes.PNG", caption="Cafes", use_column_width=True)

# Load pre-trained models
scaler = joblib.load('scaler.joblib')
kmeans = joblib.load('kmeans.joblib')

# City-to-integer mapping
city_mapping = {
    'Al Khobar': 0, 'AlUla': 1, 'Dammam': 2, 'Dhahran': 3, 'Jazan': 4, 
    'Jeddah': 5, 'Madinah': 6, 'Makkah': 7, 'Rabigh': 8, 'Riyadh': 9, 
    'Taif': 10, 'The Riyadh Province': 11
}

# Streamlit app


st.markdown("## Restaurant Clustering")
st.markdown('### خلونا نشوف الخوارزمية حقتنا :)')

# Input from the user
city = st.selectbox("Select City:", list(city_mapping.keys()))
rating = st.slider("Rating (1-5):", 1, 5)
num_reviewers = st.number_input("Number of Reviewers:", min_value=1, value=1)

def visualize_cluster(data, cluster):
    plt.figure(figsize=(8, 6))

    # Assuming kmeans cluster centers are in a 2D space for visualization
    # This is a simplified visualization assuming the first two dimensions are significant
    plt.scatter(data[:, 0], data[:, 1], c='red', label=f'Cluster {cluster}', s=100, alpha=0.6)
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='blue', marker='X', s=200, alpha=0.7, label='Centroids')

    plt.xlabel("Scaled Rating")
    plt.ylabel("Scaled Number of Reviewers")
    plt.title("Cluster Visualization")
    plt.legend()
    st.pyplot(plt)


if st.button("Predict Cluster"):
    # Prepare data for API request
    input_data = {
        "city": city,
        "rating": rating,
        "num_reviewers": num_reviewers
    }

    try:
        # Send POST request to FastAPI endpoint
        response = requests.post(api_url, json=input_data)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            result = response.json()
            cluster = result['predicted_cluster']

            # Display the cluster result
            st.success(f"The predicted cluster for {city} with rating {rating} and {num_reviewers} reviewers is: Cluster {cluster}")
            
            # Visualize the cluster (simplified without real cluster centers)
            data_final = np.array([[rating, num_reviewers]])
            visualize_cluster(data_final, cluster)
        else:
            st.error(f"Failed to get a response from the API. Status code: {response.status_code}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
