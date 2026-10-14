import os
import pathlib
import io
import pandas as pd
from PIL import Image
from huggingface_hub import hf_hub_download

BASE_DIR = pathlib.Path(r"E:\DEEPGUARD")
BENCH_DIR = BASE_DIR / "benchmark_100"
IMG_OUT_DIR = BENCH_DIR / "images"
IMG_OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load context_mismatch labels and existing images
cm_labels_path = BASE_DIR / "data" / "context_mismatch" / "labels.csv"
cm_imgs_dir = BASE_DIR / "data" / "context_mismatch" / "images"
df_cm = pd.read_csv(cm_labels_path)
existing_cm = set(os.listdir(cm_imgs_dir))

# Filter valid matching (label=0) and mismatched (label=1) pairs
df_s1_candidates = df_cm[(df_cm['label'] == 0) & (df_cm['image_id'].apply(lambda x: f"{x}.jpg" in existing_cm))].copy()
df_s4_candidates = df_cm[(df_cm['label'] == 1) & (df_cm['image_id'].apply(lambda x: f"{x}.jpg" in existing_cm))].copy()

# 2. Load CIFAKE parquet test set for Scenario 3
parquet_path = hf_hub_download(
    repo_id='dragonintelligence/CIFAKE-image-dataset',
    filename='data/test-00000-of-00001.parquet',
    repo_type='dataset'
)
df_parquet = pd.read_parquet(parquet_path)
# In CIFAKE, label 0 is FAKE (AI-generated)
df_fake_imgs = df_parquet[df_parquet['label'] == 0].reset_index(drop=True)

# Fake text claims for Scenario 2 (debunked rumors, health hoaxes, fake alerts)
fake_texts = [
    "BREAKING: Government declares complete national lockdown starting midnight tonight! All businesses and ATMs to remain closed for 14 days. Forward to save lives!",
    "Miracle cure revealed: Drinking boiled ginger with lemon water cures stage-4 respiratory infections in 48 hours without any hospital care.",
    "URGENT ALERT: Major 8.9 magnitude earthquake predicted to hit the capital city tonight at 2 AM according to satellite radar warnings.",
    "Secret leaked memo proves international space agencies are secretly controlling weather patterns to engineer artificial droughts.",
    "Shocking discovery: Consuming raw baking soda dissolves all viral infections within 10 minutes according to independent researchers.",
    "Emergency advisory: Do not answer phone calls from phone numbers starting with +9200, your bank account balance will be drained instantly upon answering.",
    "Scientists confirm artificial sweeteners contain magnetic nanotechnology chips designed to monitor citizen movements in public spaces.",
    "Breaking: Renowned national university cancels all final semester examinations and declares automatic 4.0 GPA for all registered students.",
    "Warning: Commercial 5G cellular antennas emit frequency waves that disable domestic water supply filters across urban centers.",
    "Leaked laboratory documents show major beverage brands are secretly infusing illegal addictive sedatives to increase retail consumption.",
    "Alert from healthcare ministry: Modern dental fillings contain micro-transmitters broadcasting audio frequencies to foreign intelligence servers.",
    "Urgent notification: State petroleum reserves completely exhausted, all fuel stations across the province will shut down indefinitely tomorrow morning.",
    "Shocking report: Historic bridge collapse engineered by covert international demolition teams using classified acoustic frequency resonance.",
    "Medical whistleblower reveals routine influenza immunizations contain magnetic fluid that reacts to microwave ovens in kitchens.",
    "Breaking news: Central bank issues immediate recall of all 5000-rupee currency notes, unreturned notes become void by Friday evening.",
    "Astonishing breakthrough: Ancient herbal root extract discovered in mountainous valleys restores full eyesight within three applications.",
    "Severe alert: City municipal corporation confirms drinking tap water has been contaminated with dangerous bio-engineered sleep bacteria.",
    "Secret report confirms atmospheric cloud seeding in coastal areas caused irreversible electrical grid disruptions across the northern region.",
    "Warning: Popular brand of instant cooking noodles contains chemical adhesive identical to industrial floor wax.",
    "Breaking bulletin: National meteorological authority announces temperatures will plunge below minus twenty degrees Celsius in lowlands tonight.",
    "Urgent alert: Avoid using smartphone camera applications after 9 PM as they activate infrared retina radiation that impairs night vision.",
    "Medical advisory: Raw garlic cloves worn in your socks overnight draw out heavy metal toxins and neutralize cardiovascular disease.",
    "Breaking update: Inter-provincial transport halted indefinitely after unexpected radioactive mineral deposit detected on primary highway.",
    "Confidential file: Government planning covert digital currency mandate to automatically seize 15 percent of all personal checking deposits.",
    "Emergency notice: Supermarket cooking oils blended with synthetic motor lubricants to artificially boost viscosity and boiling thresholds."
]

# Captions for Scenario 3 (AI Images)
ai_captions = [
    "Photorealistic rendering of futuristic city highway with autonomous electric transport systems.",
    "Detailed synthetic photograph of an airplane landing on runway during dramatic evening sunset.",
    "High-resolution generated portrait of domestic tabby cat resting on wooden patio chair.",
    "Artistic rendering of modern high-speed express train traveling across scenic countryside bridge.",
    "Synthetic photograph of classic red vintage sports car parked on coastal scenic viewpoint.",
    "AI generated depiction of wild brown horse galloping across expansive mountain meadow.",
    "Photorealistic synthetic image of small yellow propeller airplane flying above low cloud formations.",
    "Generated image of commercial maritime cargo ship sailing through open ocean calm waters.",
    "Detailed AI image of green passenger bus traveling along rural highway during midday sun.",
    "Synthetic render of exotic white bird perched on flowering tropical tree branch in nature.",
    "AI generated image of sleek blue convertible automobile parked outside modern residential home.",
    "Generated photographic scene of deer standing at edge of pine forest during autumn morning.",
    "Synthetic image of twin-engine private aircraft flying over snow-covered mountain peaks.",
    "Photorealistic AI rendering of maritime passenger ferry departing busy harbor terminal.",
    "Generated image of delivery truck navigating city intersection during light evening rain.",
    "Synthetic digital capture of colorful rooster standing in rural farm courtyard.",
    "AI generated render of vintage silver coupe parked on cobblestone historical street.",
    "Synthetic photograph of domestic dog running through grassy public park catching frisbee.",
    "High quality generated image of cargo freighter vessel unloading at industrial port dock.",
    "Detailed AI rendering of passenger locomotive cruising through dense alpine valley.",
    "Dramatic synthetic news photo of massive unidentified hovering spacecraft over metropolitan skyline.",
    "Photorealistic AI depiction of giant octopus emerging from asphalt street grabbing yellow taxi cab.",
    "Futuristic digital art depiction of neon-illuminated skyscraper headquarters in mega-city.",
    "Synthetic photorealistic capture of deep space research station orbiting distant ringed planet.",
    "AI generated high-detail illustration of robot assistant serving coffee in contemporary cafe."
]

records = []

# --- SCENARIO 1: Real Image + Real Text (Matching) ---
s1_subset = df_s1_candidates.iloc[:25]
for idx, (_, row) in enumerate(s1_subset.iterrows(), 1):
    src_img = cm_imgs_dir / f"{row['image_id']}.jpg"
    dst_img = IMG_OUT_DIR / f"s1_{idx:02d}.jpg"
    try:
        with Image.open(src_img) as im:
            im.convert("RGB").save(dst_img)
    except Exception as e:
        print(f"Error copying s1_{idx}: {e}")
        
    records.append({
        "id": len(records) + 1,
        "scenario": 1,
        "scenario_name": "Scenario 1: Real Image + Real Text (Matching)",
        "image_file": str(dst_img),
        "text": str(row['caption']).strip().capitalize(),
        "expected_verdict": "REAL",
        "image_score": "",
        "text_score": "",
        "mismatch_score": "",
        "final_verdict": "",
        "correct": ""
    })

# --- SCENARIO 2: Real Image + Fake Text ---
# Use 25 other real images from the dataset
s2_img_subset = df_s1_candidates.iloc[25:50]
for idx, (_, row) in enumerate(s2_img_subset.iterrows(), 1):
    src_img = cm_imgs_dir / f"{row['image_id']}.jpg"
    dst_img = IMG_OUT_DIR / f"s2_{idx:02d}.jpg"
    try:
        with Image.open(src_img) as im:
            im.convert("RGB").save(dst_img)
    except Exception as e:
        print(f"Error copying s2_{idx}: {e}")
        
    records.append({
        "id": len(records) + 1,
        "scenario": 2,
        "scenario_name": "Scenario 2: Real Image + Fake Text",
        "image_file": str(dst_img),
        "text": fake_texts[idx - 1],
        "expected_verdict": "FAKE",
        "image_score": "",
        "text_score": "",
        "mismatch_score": "",
        "final_verdict": "",
        "correct": ""
    })

# --- SCENARIO 3: Fake (AI) Image + Text ---
# Extract 20 from CIFAKE parquet, plus 5 custom AI images (UFO, Octopus, Hero banner, etc.)
for idx in range(1, 26):
    dst_img = IMG_OUT_DIR / f"s3_{idx:02d}.jpg"
    if idx == 21:
        # Custom AI UFO
        ufo_src = pathlib.Path(r"C:\Users\GA LAPTOP\.gemini\antigravity\brain\a405a7c6-d77f-4538-a4e6-e08e46e90da8\ai_generated_fake_test_1_1791655119454.jpg")
        with Image.open(ufo_src) as im:
            im.convert("RGB").save(dst_img)
    elif idx == 22:
        # Custom AI Octopus
        oct_src = pathlib.Path(r"C:\Users\GA LAPTOP\.gemini\antigravity\brain\a405a7c6-d77f-4538-a4e6-e08e46e90da8\ai_generated_fake_test_2_1791655132210.jpg")
        with Image.open(oct_src) as im:
            im.convert("RGB").save(dst_img)
    elif idx == 23:
        # Hero Banner
        hero_src = BASE_DIR / "backend" / "static" / "images" / "deepguard_hero.jpg"
        with Image.open(hero_src) as im:
            im.convert("RGB").save(dst_img)
    else:
        # From CIFAKE parquet
        parquet_idx = idx if idx < 21 else idx - 3
        raw_bytes = df_fake_imgs.iloc[parquet_idx]['image']['bytes']
        img = Image.open(io.BytesIO(raw_bytes))
        # Upscale clean thumbnail to standard resolution
        img = img.resize((224, 224), Image.Resampling.BICUBIC)
        img.save(dst_img)
        
    records.append({
        "id": len(records) + 1,
        "scenario": 3,
        "scenario_name": "Scenario 3: Fake (AI) Image + Text",
        "image_file": str(dst_img),
        "text": ai_captions[idx - 1],
        "expected_verdict": "FAKE",
        "image_score": "",
        "text_score": "",
        "mismatch_score": "",
        "final_verdict": "",
        "correct": ""
    })

# --- SCENARIO 4: Real Image + Real Text (Mismatched Context) ---
s4_subset = df_s4_candidates.iloc[:25]
for idx, (_, row) in enumerate(s4_subset.iterrows(), 1):
    src_img = cm_imgs_dir / f"{row['image_id']}.jpg"
    dst_img = IMG_OUT_DIR / f"s4_{idx:02d}.jpg"
    try:
        with Image.open(src_img) as im:
            im.convert("RGB").save(dst_img)
    except Exception as e:
        print(f"Error copying s4_{idx}: {e}")
        
    records.append({
        "id": len(records) + 1,
        "scenario": 4,
        "scenario_name": "Scenario 4: Real Image + Real Text (Context Mismatch)",
        "image_file": str(dst_img),
        "text": str(row['caption']).strip().capitalize(),
        "expected_verdict": "FAKE", # Suspicious / Context Mismatch
        "image_score": "",
        "text_score": "",
        "mismatch_score": "",
        "final_verdict": "",
        "correct": ""
    })

# Export benchmark master CSV
df_master = pd.DataFrame(records)
master_csv_path = BENCH_DIR / "deepguard_100_multimodal_benchmark.csv"
df_master.to_csv(master_csv_path, index=False)
print(f"Successfully generated {len(df_master)} benchmark samples in {master_csv_path}")
