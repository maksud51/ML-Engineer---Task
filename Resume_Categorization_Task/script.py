import glob
from langchain_community.document_loaders import PyPDFLoader
import os
import pandas as pd
import shutil
import argparse
from tqdm import tqdm
import joblib

job_classes = ["INFORMATION-TECHNOLOGY", "BUSINESS-DEVELOPMENT", "FINANCE", "ADVOCATE", "ACCOUNTANT", "ENGINEERING", 
               "CHEF", "AVIATION", "FITNESS", "SALES", "BANKING", "HEALTHCARE", "CONSULTANT", "CONSTRUCTION", 
               "PUBLIC-RELATIONS", "HR", "DESIGNER", "ARTS", "TEACHER", "APPAREL", "DIGITAL-MEDIA", "AGRICULTURE", 
               "AUTOMOBILE", "BPO"]

def directory_exist(prediction):
    path = os.path.join("output", prediction)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def check_csv_file_exist(csv_file_path):
    if not os.path.exists(csv_file_path):
        df = pd.DataFrame(columns=["file_name", "category"])
    else:
        df = pd.read_csv(csv_file_path)
    return df


def get_prediction(content,rf_classifier, vectorizer):
    try:
        new_resume_df = pd.DataFrame([content], columns=['Resume_str'])
        new_resume_tfidf = vectorizer.transform(new_resume_df['Resume_str'])
        predicted_category = rf_classifier.predict(new_resume_tfidf)
        return predicted_category[0]
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None


def load_pdf_files(directory_path):
    print("================================> Start loading files <================================")
    all_pdf_contents = []
    pdf_files = glob.glob(f"{directory_path}/*.pdf")
    for file_path in tqdm(pdf_files):
        try:
            file_name = os.path.basename(file_path)
            loader = PyPDFLoader(file_path)
            docs = loader.load()[0].page_content
            data = {
                "file_name": file_name,
                "content": docs,
                "file_path": file_path
            }
            all_pdf_contents.append(data)
        except Exception as e:
            print(f"Error in loading file {file_name}: {e}")
            continue
    print(f"=================================> Loaded Total {len(all_pdf_contents)} Files <================================")
    return all_pdf_contents

def save_prediction(all_pdf_contents,rf_classifier, vectorizer, csv_file_path="categorized_resumes.csv"):
    print("================================> Start saving predictions <================================")
    
    df = check_csv_file_exist(csv_file_path)
    df_result = []
    for pdf_content in tqdm(all_pdf_contents):
        try:
            content = pdf_content["content"]
            prediction = get_prediction(content, rf_classifier, vectorizer)
            if prediction == None:
                print(f"Prediction for file {pdf_content['file_name']} is None")
            elif prediction in job_classes and "file_name" in pdf_content and "file_path" in pdf_content:
                df_result.append({"file_name": pdf_content["file_name"], "category": prediction})
                shutil.move(pdf_content["file_path"], directory_exist(prediction))
                print(f"Prediction for file {pdf_content['file_name']} is {prediction}")
        except Exception as e:
            print(f"Error in saving prediction for file {pdf_content['file_name']}: {e}")
            continue

    if df_result:
        df_result_df = pd.DataFrame(df_result)
        df = pd.concat([df, df_result_df], ignore_index=True)
        df.to_csv(csv_file_path, index=False)
        print(f"=================================> Saved Total {len(df_result)} Predictions <================================")
    else:
        print("No predictions to save")

if __name__ == "__main__":
    rf_classifier = joblib.load('random_forest_model.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--directory', type=str, help='Path to the directory containing the PDF files.')
    args = parser.parse_args()
    directory_path = args.directory
    all_pdf_contents = load_pdf_files(directory_path)
    save_prediction(all_pdf_contents, rf_classifier, vectorizer)


#pip install -U langchain_community pypdf

"""
python script.py --directory "C:/Users/ASUS/Downloads/Maksud_final_project/Test/Test"
"""