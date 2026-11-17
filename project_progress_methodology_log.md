# Project Progress & Methodology Log: P13 — Road-Accident Hotspot Miner

**Course:** DS-311 Data Mining (Fall 2026)  
**Project Title:** P13 · Road-Accident Hotspot Miner  
**Core Research Question:** *Where do serious road accidents cluster, and which conditions (weather, light, road features) are associated with severity?*  
**Dataset:** US Accidents (2016–2023) by Moosavi et al. (Kaggle, CC BY-NC-SA 4.0)  

---

## 1. Executive Summary of Progress (M0 – M2)

| Milestone | Status | Key Deliverables Completed |
| :--- | :--- | :--- |
| **M0: Proposal & Setup** | Completed | 1-page proposal, GitHub repository initialized, memory-safe data loader built in Google Colab, deliberate stratified sampling ($N = 500,000$). |
| **M1: Data Quality & EDA** | Completed | Programmatic data dictionary, deduplication, sensor validity checks, leakage-safe train/test split ($80/20$), median/mode imputation, IQR & $z$-score outlier capping, discretization, visual EDA, and statistical hypothesis testing ($\chi^2$, Welch's $t$-test, Mann-Whitney $U$). |
| **M2: Transformations & First Models** | Completed | `ColumnTransformer` preprocessing pipeline, PCA dimensionality reduction evaluation, baseline multi-class classifiers (`DummyClassifier` vs. balanced `DecisionTreeClassifier`), and baseline FP-Growth association rules (`Conditions` $\rightarrow$ `High_Severity`). |
| **M3: Core Complete & Clustering** | Upcoming (Week 11) | Spatial clustering (DBSCAN with Haversine distance vs. $K$-Means), Random Forest multi-class severity model, and Streamlit/Gradio web app skeleton. |
| **M4: Evaluation & App Beta** | Upcoming (Week 13) | Cross-validation, explicit leakage ablation check, error analysis, and public web app deployment. |

---

## 2. Detailed Methodology & Pipeline Architecture

### Stage 1: Data Acquisition & Deliberate Sampling (M0/M1)
* **Selective Feature Loading:** Instead of loading all 46 raw columns into memory, we selected 39 domain-relevant columns (Target, Temporal/Spatial, Weather, Binary Road Flags, Light Conditions, and `Description`) to stay well below Google Colab's $12\text{ GB}$ RAM ceiling.
* **Stratified Sampling:** The raw dataset contains $7,728,394$ rows. Following course guidelines, we performed stratified random sampling (`RANDOM_SEED = 42`) on the target variable `Severity` to extract exactly $500,000$ rows.
* **Distribution Preservation:** Both the raw dataset and our $500,000$-row sample exhibit the exact same class proportions:
  * **Severity 1:** $0.87\%$
  * **Severity 2:** $79.67\%$
  * **Severity 3:** $16.81\%$
  * **Severity 4:** $2.65\%$

### Stage 2: Data-Quality Report & Leakage-Safe Preprocessing (M1)
1. **Dropping Target Leakage & Constant Columns:**
   * Dropped `End_Time`, `End_Lat`, and `End_Lng` immediately because they are recorded *after* an accident clears and directly leak accident duration/severity.
   * Dropped `Country` (constant `"US"`) and `Turning_Loop` (constant `False`, zero variance).
   * Dropped `Wind_Chill(F)` due to high missingness ($25.82\%$) and strong collinearity with `Temperature(F)` and `Wind_Speed(mph)`.
2. **Uniqueness & Deduplication:**
   * Checked exact `ID` duplicates ($0$ found).
   * Checked composite event keys (`Start_Time`, `Start_Lat`, `Start_Lng`, `Severity`) and discovered **$2,886$ repeated event duplicates** caused by multiple traffic sensors logging the same crash. Removing these left **$497,114$ unique accident records**.
3. **Consistency & String Harmonization:**
   * Standardized 24 noisy `Wind_Direction` strings (e.g., `"Calm"` $\rightarrow$ `"CALM"`, `"North"` $\rightarrow$ `"N"`, `"NNW"` $\rightarrow$ `"NW"`).
   * Consolidated 106 granular `Weather_Condition` strings into 7 clean, domain-meaningful categories (`Clear`, `Cloudy`, `Rain`, `Fog_Low_Vis`, `Snow_Ice`, `Thunderstorm`, `Windy_Dust`).
4. **Physical Validity Checks:**
   * Enforced physical sensor bounds (e.g., Temperature within $[-60^\circ\text{F}, 140^\circ\text{F}]$, Humidity within $[0\%, 100\%]$, Pressure within $[15\text{ in}, 45\text{ in}]$).
   * Flagged and replaced $3$ impossible `Temperature(F)` readings and $3$ invalid `Pressure(in)` readings with `NaN` prior to imputation.
5. **Upfront Train/Test Split & Imputation:**
   * Created an $80/20$ stratified split tag (`split == 'train'` with $397,691$ rows; `split == 'test'` with $99,423$ rows) **before** computing any summary statistics.
   * **Meaning-Based Imputation:** Filled $126,667$ missing `Precipitation(in)` values with `0.0` where `Weather_Category` was `'Clear'` or `'Cloudy'` (rain gauges often leave dry periods blank rather than logging zero).
   * **Statistical Imputation:** Imputed remaining numeric `NaN`s using the **training set median** (robust to extreme weather skewness) and categorical `NaN`s using the **training set mode**.
6. **Outlier Detection & Winsorization (Capping):**
   * Evaluated outliers using both $1.5 \times \text{IQR}$ fences and $|z| > 3$ rules computed on the training split.
   * Capped extreme sensor noise in `Temperature(F)`, `Pressure(in)`, `Wind_Speed(mph)`, and `Precipitation(in)` to the training $[P_{0.5}, P_{99.5}]$ percentiles while retaining genuine severe weather events.

### Stage 3: Exploratory Data Analysis & Hypothesis Testing (M1)
* **Feature Discretization:** Extracted `Hour`, `Weekday`, `Is_Weekend`, `Month`, `Year`, `Season`, and `Time_of_Day`. Created domain-specific weather bins (`Temp_Bin`, `Visibility_Bin`, `Wind_Bin`) and demonstrated Equal-Width vs. Equal-Frequency binning ($4$ bins) on `Temperature(F)`.
* **Hypothesis Test 1 (Categorical Association):**
  * *Test:* Chi-Square Test of Independence between `Weather_Category` and `High_Severity` (Severity $3$ or $4$).
  * *Result:* $\chi^2 = 1751.49$ ($\text{df} = 6$, $p < 0.0001$), Cramér's $V = 0.0594$. Rejected $H_0$ at $\alpha = 0.05$. Rain ($23.26\%$), Cloudy ($21.70\%$), and Thunderstorm ($20.82\%$) conditions show elevated rates of severe accidents compared to Clear weather ($17.40\%$).
* **Hypothesis Test 2 (Continuous Comparison):**
  * *Test:* Welch's Two-Sample $t$-test and Mann-Whitney $U$ test comparing `Visibility(mi)`, `Wind_Speed(mph)`, and `Temperature(F)` across Minor (Sev $1\text{–}2$) vs. Severe (Sev $3\text{–}4$) crashes.
  * *Result:* `Wind_Speed(mph)` showed both statistical and practical significance ($t = 33.34$, $p = 8.05 \times 10^{-243}$, observed gap $= +0.57\text{ mph}$ higher in severe crashes). Demonstrated why large samples ($n \approx 500\text{k}$) can yield tiny $p$-values even when the practical gap is negligible (e.g., `Visibility(mi)` Mann-Whitney $p = 1.08 \times 10^{-7}$ despite a $0.003\text{ mi}$ mean difference).

### Stage 4: Transformation, Reduction & First Baselines (M2)
* **Scikit-Learn Preprocessing Pipeline:** Built a `ColumnTransformer` that applies `StandardScaler()` to numeric features, `OneHotEncoder(handle_unknown='ignore', drop='if_binary')` to categorical features, and passes binary road flags through unchanged.
* **PCA Dimensionality Reduction Check:** Applied PCA to the 7 scaled numeric features. Capturing $\ge 90\%$ of the variance required 5 out of 7 components. **Justification for not using PCA in final models:** The numeric feature space is already low-dimensional, and transforming weather metrics into orthogonal principal components destroys domain interpretability (which is required for the interactive risk dashboard).
* **Baseline Classification:**
  1. `DummyClassifier(strategy='prior')`: Predicts the majority class (Severity 2) for all instances, establishing the floor Macro-$F_1$ score ($\approx 0.22$).
  2. `DecisionTreeClassifier(max_depth=5, class_weight='balanced')`: Accounts for the severe class imbalance and establishes our first interpretable predictive benchmark.
* **Baseline Association Rules:** Used `fpgrowth` (`min_support = 0.01`) and `association_rules` (`lift > 1.2`) on one-hot encoded conditions to discover which combinations of weather, time of day, and road infrastructure co-occur with `High_Severity == True`.

---

## 3. Bugs, Pitfalls & Issues Resolved

1. **Google Colab RAM Crash on Raw CSV Load:**
   * *Issue:* Loading the full $3\text{ GB}$ CSV (`US_Accidents_March23.csv`) with `pd.read_csv()` spikes RAM above $12\text{ GB}$ and crashes the Colab kernel.
   * *Fix:* Converted the raw dataset to `.parquet`, used `columns=selected_cols` to load only required fields, performed stratified sampling immediately, and called `del raw_df; gc.collect()` to free memory.
2. **Mixed ISO-8601 Timestamp Formats in `Start_Time`:**
   * *Issue:* Later records in the 2023 dataset include fractional nanoseconds (`.000000000`), causing standard `pd.to_datetime()` calls to fail with a `ValueError`.
   * *Fix:* Used `pd.to_datetime(df['Start_Time'], format='mixed', errors='coerce')` to safely parse all timestamps without losing rows.
3. **Silent Sensor Duplicates:**
   * *Issue:* Relying solely on `df.duplicated(subset=['ID'])` returned $0$ duplicates, masking thousands of redundant records.
   * *Fix:* Deduplicating on `['Start_Time', 'Start_Lat', 'Start_Lng', 'Severity']` caught $2,886$ duplicate sensor logs.
4. **FP-Growth Performance vs. Apriori:**
   * *Issue:* Running `apriori()` from `mlxtend` on $\approx 400,000$ training rows with $25+$ boolean columns causes severe memory slowdowns due to candidate generation.
   * *Fix:* Switched to `fpgrowth()`, which compresses transactions into an FP-tree and mines frequent itemsets in seconds on CPU.

---

## 4. Critical Watch-Outs (Viva & Rubric Checklist)

> **IMPORTANT:** Every team member must understand these points for the Week 16–17 individual viva and the M4 evaluation rubric.

### A. Data Leakage Traps (Must Be Audited!)
1. **The `Distance(mi)` Feature Trap (Action Required in M3/M4):**
   * In the US Accidents dataset, `Severity` ($1\text{–}4$) is label-encoded by the data providers based on **how much traffic delay and road closure distance (`Distance(mi)`) the accident caused**.
   * In our M1 Data Dictionary, we correctly noted `Distance(mi)` as *"Post-impact length (EDA only)"*, but in M2 Cell 1 it was included in `numeric_features` for the baseline classifier!
   * **How to handle this in M3/M4:** Including `Distance(mi)` artificially inflates classification scores because it is a post-accident measurement (target leakage). In M4 (Stage 6 requires an *"explicit leakage check"*), you should show a side-by-side comparison of your classifier **with** `Distance(mi)` vs. **without** `Distance(mi)` (using only pre-accident weather, time, and road features). This will earn full marks on the rubric's leakage check!
2. **Preprocessing Inside Pipelines Only:**
   * Never call `scaler.fit_transform(df)` or `OneHotEncoder().fit_transform(df)` on the full dataframe before splitting. Always keep `ColumnTransformer` inside a `sklearn.pipeline.Pipeline` so that during Cross-Validation (`StratifiedKFold`), scaling and encoding are fitted strictly on each training fold.
3. **Spatial Coordinates in Classification vs. Clustering:**
   * `Start_Lat` and `Start_Lng` are used for **Spatial Clustering (DBSCAN / $K$-Means)**, not as raw unbinned inputs to the global decision tree (unless converted into regional cluster IDs or hotspot risk zones).

### B. Evaluation Metric Trap (Why Accuracy is Wrong)
* Because Severity 2 makes up **$79.67\%$** of the dataset, a useless model that predicts "Severity 2" every single time achieves **$79.67\%$ Accuracy**.
* Always report **Macro-$F_1$** (`f1_score(y_test, y_pred, average='macro')`), per-class Precision/Recall, and a **Normalized Confusion Matrix**. Macro-$F_1$ weights all 4 severity classes equally, heavily penalizing models that ignore rare Severity 1 ($0.87\%$) and Severity 4 ($2.65\%$) crashes.

### C. Spatial Clustering Watch-Outs (For M3 DBSCAN)
* **Haversine Distance & Radians:** Standard Euclidean distance distorts geographical distances because lines of longitude converge toward the poles. When running `DBSCAN(metric='haversine')` in scikit-learn:
  1. Coordinates **must** be converted from degrees to **radians** using `np.radians(df[['Start_Lat', 'Start_Lng']])`.
  2. The `eps` parameter must also be in radians: divide your target search radius in kilometers by Earth's radius ($R \approx 6371.0088\text{ km}$). For example, a $5\text{ km}$ radius is $\text{eps} = 5 / 6371.0088 \approx 0.0007848\text{ radians}$.
* **Memory Limit in DBSCAN:** Running `DBSCAN` with `metric='haversine'` and `algorithm='ball_tree'` on $500,000$ points at once can exceed $12\text{ GB}$ RAM if `eps` is too large. Filter to **serious accidents** (`Severity` $\in \{3, 4\}$) or focus on 1–2 high-density states (as permitted by the P13 project guide) when running dense spatial clustering.

### D. Pakistan Context Watch-Out (Required for Final Report)
* The P13 project guide explicitly states: *"Watch out: US data; the report should discuss what similar data exists or is missing for Pakistan."*
* Be prepared to include a ½-page discussion in your report comparing this dataset to Pakistan's road safety landscape:
  * **Existing Sources in Pakistan:** National Highways & Motorway Police (NHMP) daily incident logs, provincial Rescue 1122 emergency dispatch records (Punjab, KP), and Pakistan Bureau of Statistics (PBS) annual traffic summaries.
  * **What is Missing in Pakistan:** Lack of open, geocoded ($\text{Lat}/\text{Lng}$) crash-level datasets; absence of automated join pipelines linking accidents to Pakistan Meteorological Department (PMD) weather stations; and lack of standardized OpenStreetMap (OSM) road-infrastructure tags (`Crossing`, `Junction`, `Traffic_Signal`) integrated into police FIRs/crash reports.

---

## 5. Future Work & Next Steps (M3 $\rightarrow$ Final Submission)

### Milestone 3 (Week 11): Core Complete, Clustering & App Skeleton
- [ ] **Spatial Hotspot Mining (Intermediate):**
  - Implement `DBSCAN(metric='haversine', algorithm='ball_tree')` on radian coordinates of severe accidents.
  - Run a parameter sensitivity study showing how varying `eps` (e.g., $1\text{ km}$, $3\text{ km}$, $5\text{ km}$) and `min_samples` (e.g., $15, 30, 50$) changes the number of clusters and noise points.
  - Compare DBSCAN against standard $K$-Means (highlighting why $K$-Means struggles with irregular highway geometry and cannot discard background noise).
- [ ] **Intermediate Classification (Random Forest):**
  - Train a `RandomForestClassifier(class_weight='balanced_subsample', random_state=42)` inside the preprocessing pipeline and compare its Macro-$F_1$ against the M2 `DummyClassifier` and `DecisionTreeClassifier`.
- [ ] **Association Rules Refinement:**
  - Include discretized bins (`Temp_Bin`, `Visibility_Bin`, `Wind_Bin`) from M1 in the FP-Growth transaction matrix and interpret the top 5 highest-lift rules in plain English.
- [ ] **Web App Skeleton:**
  - Build a basic Streamlit (`app.py`) page with an interactive map (`folium` or `pydeck`) displaying the top spatial hotspot clusters.

### Milestone 4 (Week 13): Full Evaluation & Public Beta
- [ ] **Cross-Validation & Tuning:** Run 5-fold `StratifiedKFold` cross-validation on the classification pipelines and tune hyperparameters (`max_depth`, `min_samples_leaf`, `n_estimators`).
- [ ] **Explicit Leakage Check:** Document the `Distance(mi)` ablation experiment (Macro-$F_1$ with vs. without post-accident distance).
- [ ] **Error Analysis & Cluster Profiling:** Name the discovered spatial clusters (e.g., *"Los Angeles I-405 Commuter Corridor"*, *"Miami I-95 Rain/Interchange Cluster"*) and analyze false positives/negatives from the classifier.
- [ ] **Public Deployment:** Deploy the Streamlit app to Streamlit Community Cloud (or Gradio to Hugging Face Spaces) with:
  1. Interactive Hotspot Map
  2. Condition Risk Selector (user selects weather, time, and road features $\rightarrow$ model outputs predicted severity risk)
  3. Association Rule Explorer

### Optional Expert Module (Week 15 Bonus — Choose One if Time Permits)
- [ ] **Option A (Temporal-Spatial Shift):** Run `HDBSCAN` across different seasons (`Winter` vs. `Summer`) to visualize how severe accident hotspots shift geographically during winter snow/ice.
- [ ] **Option B (Text Mining):** Apply `TfidfVectorizer` to the free-text `Description` column to extract incident sub-types (e.g., *"multi-vehicle"*, *"jackknifed semi"*, *"contraflow"*, *"hydroplaned"*) and test whether text features boost Macro-$F_1$.