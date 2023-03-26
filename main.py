# Step 1: Load libraries
import pickle
from utils.data_processing import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer

# Step 2: Load saved model
with open('knn_eng.sav', 'rb') as f:
    my_model = pickle.load(f)

# Step 3: Make predictions
new_data = """The young man drove the rich on the highway.Lift them up the police stationPunch the police, the nose is broken-the little finger is broken.|The young man lifted up the police station. 1 asked the reason to set the rubber cone on the expressway. After driving until the car is damaged The police clarified it was punching the nose. The little hand of the right hand (28 Nov 64) at 8:00 pm, Police Lieutenant Aran Chavanon, Deputy Deputy Chief of Staff, Express Traffic Control Center, Kor Kor. Yesterday (27 Nov 64) at approximately 23.00 hrs. While he performed his duties at the Expressway 1 police station, he had a name later. The police station by informing that at 8:00 pm, he drove a white Honda Jazz brand on the Port 2 Expressway heading to Bang Na. Before Chonchon Rich Tang Which is located on the traffic surface Until causing the car to be damaged He therefore tried to clarify the cause of the rubber cone and explain the law to the police. Revealed that after that, Mr. Sittichok has argued. Then became anger and resentment He therefore informed him to contact the traffic officer during office hours. But the man spoke loudly And bring the phone to take pictures So he raised his hand to close the camera and told him to calm down. Mr Sitthok used the fist to fight into his face until the nosebleeds flowed. While the group of male friends who came with the attitude to attack, Pol. After Mr Sitthok caused the incident, he drove out of the police station. He was sent to treatment at the police hospital. The doctor sewed the wounds around the nose and diagnosed that the right hand bone, right hand, broken. Revealed that After the incident, he learned that while he was treated Mr Sittichok, the parties, reported to condone themselves to assault with the investigating officers at the Port Police Station. By claiming that the side of the police tried to draw a gun And punching Mr. Sittok first He therefore reported to the Port Police Station. Initially, in the matter of reporting charges At this time, it is not possible to notify. Since the inquiry official is in the process of collecting evidence Including requesting images from CCTV on the expressway To be considered
Grandma led the grandchildren to report more.The monk uses a rocket.Claiming 4 herds of hernia|Grandmother, Primary 5, reported to the offense. While another grade 6 grandmother came out, revealed that the grandchildren were hit 2 times, including 4 children being done, expected to have more victims (29 Nov 64) officials from social development and human security. , The police suppressing human trafficking And related agencies Traveled to meet Phra Sombat, 40 years old, Abbot of Pradu Temple To investigate the facts After becoming news Forcing grade 3 students and grade 5, spinning the genitals until the orgasm And later a grade 1 student came out and revealed that 5 years ago was hit by the abbot Doing the same. Most recently, there was another grade 6 grandmother to report. After the grandchildren told The abbot has done the same 2 times. In total, there were 4 children being done to report 2 monks and expected to add more victims. Because of this kind of way for many years, Mr Thongthip Phu Si to specialist in specialized social developers The Office of Social Development and Human Security states that there must be a thorough investigation. That the story that the child told Is it true? There will be many agencies together. Which must be fair to all parties Especially the abbot Who still denied that they did not act Confessed only that To massage the genitals only. Later, Grandma, the 5th grade boy who came out to give information to the first teacher. Traveled to meet the inquiry official Confirmed to condone the abbot or His Majesty's abundance While the children of Primary 5 were found, the inquiry official was 58 years old. That was abbot Do the same 2 times but do not dare to tell Grandma. Until the same school students revealed Therefore dare to tell Grandma After this, it will be allowed to follow the legal procedures. To protect your own grandchildren"""

preprocessing_data = preprocessing(new_data)

tfidf  = TfidfVectorizer(
  stop_words = 'english',                      # ป้อนรายการคำ Stop words 
  ngram_range = (1,2),                         # อยากวิเคราะห์แบบ 2 คำ ติดกัน
  min_df = 5,                                  # ขั้นต่ำของ Doc Freq ของ Term
  max_df = 0.95,
  max_features = 50000,
  norm='l2'
)

print("preprocessing_data", preprocessing_data)

vector_text = tfidf.fit_transform([preprocessing_data])
print(vector_text)
# predictions = my_model.predict(vector_text)

# print(predictions)
