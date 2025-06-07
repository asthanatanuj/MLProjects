import pandas as pd
import numpy
import sklearn
import lightgbm
from sklearn.linear_model import LinearRegression
from openai import OpenAI

original_data = r'C:\Users\astha\Downloads\ai_job_dataset.csv'

client = OpenAI(api_key='secret')


field_input = str(input('Input your specified field: '))
country_input = str(input('Input the country you live in: '))
salary_input = str(input('Input any salary prefrences. Type "NA" if not applicable: '))
education_input = str(input("Input your education level (Bachelor, Master, or PhD): "))
experience_input = str(input('Input your experience level. EN for entry level, MI for mid level, SE for senior level, or EX for executive level: '))
date_input = str(input("What is the date today? Enter in (D)D/(M)M/YYYY format: "))
remote_input = str(input("Would you like to work remote? Type '100%' if fully remote and any range in between for hybrid: "))
skills_input = str(input("what skills do you have?: "))

user_inputs = {
    "field" : field_input,
    "country" : country_input,
    "salary" : salary_input,
    "education" : education_input,
    "experience" : experience_input,
    "date" : date_input,
    "remote" : remote_input,
    "skills" : skills_input
}


train_df = pd.read_csv(original_data)




def generate_prompt(user_input, train_df):
    prompt = f"""You are a career recommendation engine. Based on the user's inputs and the job dataset below use machine learning to recommend the top 10 matching jobs.

    User Inputs:
    - Field: {field_input}
    - Country: {country_input}
    - Salary Preference: {salary_input}
    - Education: {education_input}
    - Experience Level: {experience_input}
    - Date: {date_input}
    - Remote Preference: {"Yes" if remote_input == 'Y' else "No"}
    - Skills: {skills_input}
    
    Job Dataset (columns: job_title, salary_usd, experience, education, company_location, remote_ratio, required_skills):
    {train_df.to_string(index=False)}

Please recommend the top 3 most relevant job titles for this user and explain why."""
    return prompt


prompt = generate_prompt(user_inputs, train_df)

response = client.chat.completions.create(
    model = "gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print(response.choices[0].message.content)
