import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import plotly.express as px

# -----------------------
# Utility / config
# -----------------------
EXPECTED_COLUMNS = [
    'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
    'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'
]

MODEL_FILES = [
    ('Decision Trees', 'tree.pkl'),
    ('Logistic Regression', 'LogisticRegression.pkl'),
    ('Random Forest', 'RandomForest.pkl'),
    ('Support Vector Machine', 'SVM.pkl'),
    ('GridRandom', 'gridrf.pkl')
]


def get_binary_file_downloader_html(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions CSV</a>'


# -----------------------
# Validation helpers
# -----------------------
def is_integer_like(series):
    try:
        pd.to_numeric(series.dropna(), errors='raise')
        # check if all ints
        return np.all(np.equal(np.mod(pd.to_numeric(series.dropna()), 1), 0))
    except Exception:
        return False


def find_invalid_rows(df):
    """
    Returns a dict column -> list of (index, value) for invalid entries
    based on heuristics for each expected column.
    """
    invalid = {}
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            invalid[col] = [('__missing_column__', '__missing__')]
            continue

        series = df[col]

        # If any NaNs -> invalid
        nan_mask = series.isna()
        bad_indices = list(series[nan_mask].index)
        invalid_examples = [(int(i), series.iloc[i]) for i in bad_indices]

        # Column-specific heuristics
        if col == 'Age':
            # numeric & 0-150
            coerced = pd.to_numeric(series, errors='coerce')
            mask = coerced.isna() | (coerced < 0) | (coerced > 150)
            for i in series[mask].index:
                invalid_examples.append((int(i), series.iloc[i]))

        elif col == 'Sex':
            # Accept numeric 0/1 or strings Male/Female (case-insensitive)
            def ok_sex(v):
                if pd.isna(v): return False
                if isinstance(v, (int, np.integer)) and v in (0, 1): return True
                s = str(v).strip().lower()
                return s in ('male', 'female', 'm', 'f', '0', '1')
            for i, v in series.items():
                if not ok_sex(v):
                    invalid_examples.append((int(i), v))

        elif col == 'ChestPainType':
            # Accept strings containing keywords or small integer codes
            valid_keywords = ['typical', 'atypical', 'non', 'asymptomatic', 'non-anginal']
            for i, v in series.items():
                if pd.isna(v):
                    continue
                if isinstance(v, (int, np.integer)) and (0 <= int(v) <= 10):
                    # numeric allowed — accept (can't be sure mapping)
                    continue
                s = str(v).strip().lower()
                if not any(k in s for k in valid_keywords) and not s.isdigit():
                    invalid_examples.append((int(i), v))

        elif col == 'RestingBP':
            c = pd.to_numeric(series, errors='coerce')
            mask = c.isna() | (c < 0) | (c > 400)
            for i in series[mask].index:
                invalid_examples.append((int(i), series.iloc[i]))

        elif col == 'Cholesterol':
            c = pd.to_numeric(series, errors='coerce')
            mask = c.isna() | (c < 0) | (c > 1000)
            for i in series[mask].index:
                invalid_examples.append((int(i), series.iloc[i]))

        elif col == 'FastingBS':
            # Accept numeric 0/1 or string variants
            def ok_fb(v):
                if pd.isna(v): return False
                if isinstance(v, (int, np.integer)) and v in (0, 1): return True
                s = str(v).strip().lower()
                return s in ('0', '1', '<= 120 mg/dl', '> 120 mg/dl', 'no', 'yes', 'false', 'true')
            for i, v in series.items():
                if not ok_fb(v):
                    invalid_examples.append((int(i), v))

        elif col == 'RestingECG':
            # Accept numeric 0/1/2 or keyword matches
            valid_keywords = ['normal', 'st-t', 'st-t wave', 'left', 'hypertrophy']
            for i, v in series.items():
                if pd.isna(v): continue
                if isinstance(v, (int, np.integer)) and int(v) in (0, 1, 2): continue
                s = str(v).strip().lower()
                if not any(k in s for k in valid_keywords) and not s.isdigit():
                    invalid_examples.append((int(i), v))

        elif col == 'MaxHR':
            c = pd.to_numeric(series, errors='coerce')
            mask = c.isna() | (c < 30) | (c > 250)
            for i in series[mask].index:
                invalid_examples.append((int(i), series.iloc[i]))

        elif col == 'ExerciseAngina':
            def ok_ex(v):
                if pd.isna(v): return False
                if isinstance(v, (int, np.integer)) and int(v) in (0, 1, 8): return True
                s = str(v).strip().lower()
                return s in ('yes', 'no', 'y', 'n', '1', '0', '8')
            for i, v in series.items():
                if not ok_ex(v):
                    invalid_examples.append((int(i), v))

        elif col == 'Oldpeak':
            c = pd.to_numeric(series, errors='coerce')
            mask = c.isna() | (c < 0) | (c > 20)
            for i in series[mask].index:
                invalid_examples.append((int(i), series.iloc[i]))

        elif col == 'ST_Slope':
            # Accept numeric 0/1/2 or strings upsloping/flat/downsloping
            valid_keywords = ['upslop', 'flat', 'downslop']
            for i, v in series.items():
                if pd.isna(v): continue
                if isinstance(v, (int, np.integer)) and int(v) in (0, 1, 2): continue
                s = str(v).strip().lower()
                if not any(k in s for k in valid_keywords) and not s.isdigit():
                    invalid_examples.append((int(i), v))

        # remove duplicates and limit to 5 examples
        seen = set()
        examples = []
        for ex in invalid_examples:
            if (ex[0], str(ex[1])) not in seen:
                examples.append(ex)
                seen.add((ex[0], str(ex[1])))
            if len(examples) >= 5:
                break

        if examples:
            invalid[col] = examples

    return invalid


# -----------------------
# Auto-clean helpers
# -----------------------
def auto_clean_dataframe(df):
    """
    Attempt to fix common issues:
    - BOM/leading spaces
    - header case mismatches (age -> Age)
    - convert common text values to numeric encodings used by the app UI
      (Sex: Male->0 Female->1, ExerciseAngina: Yes->1 No->0, etc.)
    Returns cleaned_df, report (string)
    """
    report_lines = []
    df = df.copy()
    # fix headers
    df.columns = df.columns.str.strip()
    # Try to rename columns by lowercase matching
    colmap = {c.lower(): c for c in EXPECTED_COLUMNS}
    df = df.rename(columns=lambda x: colmap.get(str(x).strip().lower(), x))

    # Now apply per-column conversions if they seem stringy
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].apply(lambda v: 0 if str(v).strip().lower() in ('male', 'm', '0') else (1 if str(v).strip().lower() in ('female', 'f', '1') else v))
        report_lines.append('Sex: mapped Male/Female -> 0/1 when possible.')

    if 'ChestPainType' in df.columns:
        def map_chest(v):
            s = str(v).strip().lower()
            if s == 'typical angina' or 'typical' in s: return 0
            if s == 'atypical angina' or 'atypical' in s: return 1
            if 'non' in s or 'non-anginal' in s: return 2
            if 'asymptomatic' in s: return 3
            # if numeric string, convert
            try:
                return int(float(s))
            except:
                return v
        df['ChestPainType'] = df['ChestPainType'].apply(map_chest)
        report_lines.append('ChestPainType: mapped common strings -> 0..3 (typical->0, atypical->1, non-anginal->2, asymptomatic->3)')

    if 'FastingBS' in df.columns:
        df['FastingBS'] = df['FastingBS'].apply(lambda v: 1 if str(v).strip().lower() in ('> 120 mg/dl', '>120 mg/dl', '>120', '1', 'true', 'yes') else (0 if str(v).strip().lower() in ('<= 120 mg/dl', '<=120 mg/dl', '<=120', '0', 'false', 'no') else v))
        report_lines.append('FastingBS: normalized common text to 0/1 when possible.')

    if 'RestingECG' in df.columns:
        def map_ecg(v):
            s = str(v).strip().lower()
            if 'normal' in s: return 0
            if 'st' in s or 't' in s or 'abnormal' in s: return 1
            if 'left' in s or 'ventricular' in s or 'hypertrophy' in s: return 2
            try:
                return int(float(s))
            except:
                return v
        df['RestingECG'] = df['RestingECG'].apply(map_ecg)
        report_lines.append('RestingECG: attempted text->0/1/2 mapping.')

    if 'ExerciseAngina' in df.columns:
        df['ExerciseAngina'] = df['ExerciseAngina'].apply(lambda v: 1 if str(v).strip().lower() in ('yes', 'y', '1', 'true') else (0 if str(v).strip().lower() in ('no', 'n', '0', 'false') else v))
        report_lines.append('ExerciseAngina: normalized Yes/No -> 1/0 when possible.')

    if 'ST_Slope' in df.columns:
        def map_slope(v):
            s = str(v).strip().lower()
            if 'ups' in s: return 0
            if 'flat' in s: return 1
            if 'down' in s: return 2
            try:
                return int(float(s))
            except:
                return v
        df['ST_Slope'] = df['ST_Slope'].apply(map_slope)
        report_lines.append('ST_Slope: mapped Upsloping/Flat/Downsloping -> 0/1/2 when possible.')

    return df, '\n'.join(report_lines) if report_lines else 'No auto-changes applied.'


# -----------------------
# Prediction helper
# -----------------------
def predict_all_models(input_df):
    results = {}
    for name, fname in MODEL_FILES:
        try:
            model = pickle.load(open(fname, 'rb'))
        except Exception as e:
            results[name] = {'error': f'Cannot load model file {fname}: {e}'}
            continue
        try:
            preds = model.predict(input_df[EXPECTED_COLUMNS])
            results[name] = {'predictions': preds}
        except Exception as e:
            results[name] = {'error': f'Prediction failed: {e}'}
    return results


# -----------------------
# Streamlit app layout
# -----------------------
st.set_page_config(layout="wide")
st.title("Heart Disease Predictor — CSV Validator + Cleaner + Predictor")

tab1, tab2, tab3 = st.tabs(['Single Predict', 'Bulk CSV Predict (validate & fix)', 'Model Info / Help'])

# -----------------------
# TAB 1: single input
# -----------------------
with tab1:
    st.header("Single (manual) prediction")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age (years)", min_value=0, max_value=150, value=45)
        sex = st.selectbox("Sex", ["Male", "Female"])
        chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    with col2:
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300, value=120)
        cholesterol = st.number_input("Serum Cholesterol (mm/dl)", min_value=0, value=200)
        fasting_bs = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])
    with col3:
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
        max_hr = st.number_input("Max Heart Rate", min_value=30, max_value=250, value=150)
        exercise_angina = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
        oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=20.0, value=1.0)
        st_slope = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])

    # map these to numeric encodings consistent with app UI
    sex_v = 0 if sex == "Male" else 1
    chest_v = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(chest_pain)
    fasting_v = 1 if fasting_bs == "> 120 mg/dl" else 0
    resting_ecg_v = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg)
    exercise_v = 1 if exercise_angina == "Yes" else 0
    slope_v = ["Upsloping", "Flat", "Downsloping"].index(st_slope)

    input_row = pd.DataFrame([{
        'Age': age, 'Sex': sex_v, 'ChestPainType': chest_v, 'RestingBP': resting_bp,
        'Cholesterol': cholesterol, 'FastingBS': fasting_v, 'RestingECG': resting_ecg_v,
        'MaxHR': max_hr, 'ExerciseAngina': exercise_v, 'Oldpeak': oldpeak, 'ST_Slope': slope_v
    }])

    if st.button("Predict (single)"):
        st.subheader("Prediction")
        # try to predict using all models
        preds = predict_all_models(input_row)
        for mname, res in preds.items():
            st.markdown("---")
            st.write(f"Model: **{mname}**")
            if 'error' in res:
                st.error(res['error'])
            else:
                p = int(res['predictions'][0])
                if p == 0:
                    st.success("🟢 LOW RISK – No heart disease detected.")
                else:
                    st.error("🔴 HIGH RISK – Possible heart disease detected. Consult a doctor.")


# -----------------------
# TAB 2: bulk CSV
# -----------------------
with tab2:
    st.header("Bulk CSV validation → show incorrect fields → clean → predict")
    st.markdown("Upload a CSV. The app will show which columns are missing, and for each column show example invalid rows (index & value). You can optionally auto-clean common issues and then predict.")

    uploaded_file = st.file_uploader("Upload CSV file (must have header row)", type=['csv'])

    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file, encoding='utf-8-sig')
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")
            df_raw = None

        if df_raw is not None:
            st.subheader("Detected columns in your file")
            st.write(list(df_raw.columns))

            # quick header normalization display
            df_headers_clean = df_raw.copy()
            df_headers_clean.columns = df_headers_clean.columns.str.strip()
            st.write("Stripped header names (whitespace removed):")
            st.write(list(df_headers_clean.columns))

            # Show missing/extra columns
            missing = [c for c in EXPECTED_COLUMNS if c not in df_headers_clean.columns]
            extra = [c for c in df_headers_clean.columns if c not in EXPECTED_COLUMNS]
            if missing:
                st.warning(f"Missing expected columns: {missing}")
            else:
                st.success("All expected columns are present (by name).")

            if extra:
                st.info(f"Extra/unexpected columns detected: {extra}")

            # run detailed per-column invalid value detection
            st.subheader("Column-level invalid value report (examples)")
            invalid = find_invalid_rows(df_headers_clean)
            any_invalid = False
            for col in EXPECTED_COLUMNS:
                if col not in df_headers_clean.columns:
                    st.error(f"Column **{col}** is MISSING.")
                    any_invalid = True
                elif col in invalid:
                    any_invalid = True
                    st.markdown(f"**{col}** — invalid examples (index, value):")
                    examples = invalid[col]
                    # show in a small table
                    ex_df = pd.DataFrame(examples, columns=['index', 'value'])
                    ex_df.index = range(1, len(ex_df) + 1)
                    st.table(ex_df)
                else:
                    st.success(f"{col} — looks OK (basic checks).")

            if not any_invalid:
                st.success("No obvious invalid values found (basic heuristics).")

            # show preview data
            st.subheader("Data preview (first 10 rows)")
            st.write(df_headers_clean.head(10))

            # Offer auto-clean option
            st.markdown("---")
            st.subheader("Auto-clean common issues (optional)")
            st.write("This will attempt to: fix header case/whitespace, normalize Sex/ExerciseAngina/FastingBS strings -> 0/1, map common ChestPainType and ST_Slope strings -> numeric codes used by this app (typical->0, atypical->1, non-anginal->2, asymptomatic->3; Upsloping->0, Flat->1, Downsloping->2).")
            do_clean = st.checkbox("Auto-clean and show cleaned preview")

            cleaned_df = None
            clean_report = ''
            if do_clean:
                cleaned_df, clean_report = auto_clean_dataframe(df_headers_clean)
                st.subheader("Auto-clean report")
                st.write(clean_report)
                st.subheader("Cleaned preview (first 10 rows)")
                st.write(cleaned_df.head(10))

                # re-run invalid checks on cleaned
                st.subheader("Validation after auto-clean")
                invalid_after = find_invalid_rows(cleaned_df)
                any_invalid_after = False
                for col in EXPECTED_COLUMNS:
                    if col not in cleaned_df.columns:
                        st.error(f"Column **{col}** is STILL MISSING after auto-clean.")
                        any_invalid_after = True
                    elif col in invalid_after:
                        any_invalid_after = True
                        st.markdown(f"**{col}** — still invalid examples (index, value):")
                        examples = invalid_after[col]
                        ex_df = pd.DataFrame(examples, columns=['index', 'value'])
                        ex_df.index = range(1, len(ex_df) + 1)
                        st.table(ex_df)
                    else:
                        st.success(f"{col} — looks OK after auto-clean.")
                if not any_invalid_after:
                    st.success("Auto-clean resolved all basic validation issues (heuristic checks).")

            # If cleaned and valid (or original valid), allow prediction
            can_predict = False
            final_df = None
            if do_clean and cleaned_df is not None:
                # check presence of expected columns and no invalid heuristics
                invalid_after = find_invalid_rows(cleaned_df)
                if all(col in cleaned_df.columns for col in EXPECTED_COLUMNS) and len(invalid_after) == 0:
                    can_predict = True
                    final_df = cleaned_df.copy()
                else:
                    st.warning("Auto-clean did not fully fix all issues. Fix the flagged values and try again.")
            else:
                # use original if valid
                inv_orig = find_invalid_rows(df_headers_clean)
                if all(col in df_headers_clean.columns for col in EXPECTED_COLUMNS) and len(inv_orig) == 0:
                    can_predict = True
                    final_df = df_headers_clean.copy()
                else:
                    st.info("You can either fix the CSV outside this app, or try the auto-clean option above.")

            if can_predict and final_df is not None:
                st.markdown("---")
                st.subheader("Ready to predict — choose model(s) and run")
                chosen_models = st.multiselect("Select models to run", [m for m, _ in MODEL_FILES], default=[m for m, _ in MODEL_FILES])
                if st.button("Run predictions on cleaned data"):
                    # prepare input: ensure numeric types
                    for col in EXPECTED_COLUMNS:
                        # attempt numeric conversion where appropriate
                        if col in ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope', 'Sex', 'ChestPainType', 'RestingECG']:
                            try:
                                final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
                            except:
                                pass

                    # run predictions per chosen model
                    results = {}
                    for mname, fname in MODEL_FILES:
                        if mname not in chosen_models:
                            continue
                        try:
                            model = pickle.load(open(fname, 'rb'))
                        except Exception as e:
                            st.error(f"Cannot load model {fname}: {e}")
                            continue
                        try:
                            preds = model.predict(final_df[EXPECTED_COLUMNS])
                        except Exception as e:
                            st.error(f"Prediction failed for model {mname}: {e}")
                            continue
                        final_df[f'Pred_{mname}'] = preds
                        st.success(f"Predictions added for model: {mname}")

                    st.subheader("Prediction results (first 20 rows)")
                    st.write(final_df.head(20))

                    st.markdown(get_binary_file_downloader_html(final_df), unsafe_allow_html=True)


# -----------------------
# TAB 3: Info / Model performance
# -----------------------
with tab3:
    st.header("Model info & quick chart")
    st.write("Available models (files expected in same folder):")
    for mname, fname in MODEL_FILES:
        st.write(f"- {mname}: file `{fname}`")

    data = {
        'Decision Trees': 80.97,
        'Logistic Regression': 85.86,
        'Random Forest': 84.23,
        'Support Vector Machine': 84.22,
        'GridRF': 83.75
    }
    df_perf = pd.DataFrame(list(data.items()), columns=['Models', 'Accuracies'])
    fig = px.bar(df_perf, x='Models', y='Accuracies')
    st.plotly_chart(fig)

    st.markdown("---")
    st.subheader("Notes & tips")
    st.write("""
    - Column names are **case-sensitive** and must exactly match the expected list shown above.
    - If your models were trained with different numeric encodings for categorical columns (e.g. ChestPainType coded as 3,8,1,2), DO NOT use the auto-clean mappings — instead supply the CSV with the exact numeric codes used in training.
    - The app uses heuristics to detect suspicious values (NaN, out-of-range numbers, unknown strings). These heuristics are helpful but not foolproof.
    - If you want, paste the header row of your CSV here and I will show exactly how to rename it for compatibility.
    """)

