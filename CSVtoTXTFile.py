import csv
import pandas as pd
import json
import ast
import os
import math

# Open CSV file to read CSV, note: reading and write file should be under "with"

# file = "yelp_academic_dataset_business_total_dataset_small"
file = "textFiles/linkedin_data"

df = pd.read_csv("../data/updated_fixed_linkedin.csv")
df = df.where(pd.notnull(df), None)

print(os.path.abspath(os.getcwd()))

# print(json.dumps(df, indent=4))

tempDict = {}

for i, row in df.iterrows():
    tempDict[i] = row

counter = 0

final = {}

# for k, v in tempDict.items():
#     final[v[0]] = ast.literal_eval(v[1])

# for k, v in tempDict.items():
#     final[v[0]] = {
#         "id": v[0],
#         "name": v[1],
#         "years of experience": v[2],
#         "current_job": v[3],
#         "current_job_company_id": v[4],
#         "current_job_id": v[5],
#         "current_company": v[6],
#         "current_company_id": v[7],
#         "education_degree": v[8],
#         "education_degree_type": v[9],
#         "education_ids": v[10],
#         "skills": v[11],
#         "skills_ids": v[12],
#         "industries": v[13],
#         "total_jobs_history": v[14],
#         "total_companies_history": v[15],
#         "past_jobs": v[16],
#         "past_jobs_ids": v[17],
#         "past_companies": v[18],
#         "past_education_degree": v[19],
#         "past_education_type": v[20]
#     }




# user-skills - u2s
# user-experience - u2exp
# user-education - u2edu

# 0 = id


u2skills = open(file + "_u2s.txt", "w", encoding="utf-8")
u2pos = open(file + "_u2pos.txt", "w", encoding="utf-8")
u2exp = open(file + "_u2exp.txt", "w", encoding="utf-8")
u2edu = open(file + "_u2edu.txt", "w", encoding="utf-8")
u2pos_history = open(file + "_u2posHistory.txt", "w", encoding="utf-8")
u2company = open(file + "_u2company.txt", "w", encoding="utf-8")
u2skill_Str = open(file + "_u2skillStr.txt", "w", encoding="utf-8")
u2exp_Str = open(file + "u2expStr.txt", "w", encoding="utf-8")

counter = 0

for k, v in tempDict.items():
    print("\n\n\n")
    # if counter != 100:
    # print(v)
    user_id = str(v["id"])
    username = str(v["name"])
    user_position_id = str(v["current_job_id"])
    user_current_company = str(v["current_company_id"])
    user_current_job_id = str(v["current_job_company_id"])
    user_jobs = str(v["total_jobs_history"])
    user_skills = str(list(ast.literal_eval((v["skills_ids"]))))
    user_years_of_experience = str(v["years of experience"])
    user_education = str(v["education_ids"])
    if v["total_jobs_history"]:
        total_jobs_history = str(list(ast.literal_eval(v["total_jobs_history"])))

    if v["past_jobs_ids"]:
        past_position_ids = str(list(ast.literal_eval(v["past_jobs_ids"])))

    if v["total_companies_history"]:
        past_companies = str(list(ast.literal_eval(v["total_companies_history"])))

    if v["current_job"]:
        current_job = str(v["current_job"])

    if len(total_jobs_history) > 1:
        # u2pos.write(str(k) + "`t" + user_id + "`t" + user_position_id + "`t" + str(1) + "\n")
        # u2exp.write(str(k) + "`t" + user_id + "`t" + total_jobs_history + "`t" + str(1) + "\n")
        # u2skills.write(str(k) + "`t" + user_id + "`t" + user_skills + "`t" + str(1) + "\n")
        # u2edu.write(str(k) + "`t" + user_id + "`t" + user_education + "`t" + str(1) + "\n")
        # u2pos_history.write(str(k) + "`t" + user_id + "`t" + past_position_ids + "`t" + str(1) + "\n")
        # u2company.write(str(k) + "`t" + user_id + "`t" + past_companies + "`t" + str(1) + "\n")
        # u2skill_Str.write(str(k) + "`t" + user_id + "`t" + past_companies + "`t" + str(1) + "\n")
        # u2exp_Str.write(str(k) + "`t" + user_id + "`t" + current_job + "`t" + past_companies + "`t" + str(1) + "\n")
        u2pos.write(user_id + " " + user_position_id + " " + str(1) + "\n")

        print("\n\n", type(total_jobs_history))
        print(type(user_skills))
        print(type(user_education))
        print(type(past_position_ids))
        print(type(past_companies))

        print(total_jobs_history)
        print(user_skills)
        print(user_education)
        print(past_position_ids)
        print(past_companies)


        if isinstance(total_jobs_history, str):
            total_jobs_history = ast.literal_eval(total_jobs_history)

        if isinstance(user_skills, str):
            user_skills = ast.literal_eval(user_skills)

        if isinstance(user_education, str):
            user_education = ast.literal_eval(user_education)

        if isinstance(past_position_ids, str):
            past_position_ids = ast.literal_eval(past_position_ids)

        if isinstance(past_companies, str):
            past_companies = ast.literal_eval(past_companies)

        print(total_jobs_history)
        counter = 1
        if len(total_jobs_history) > 4:
            total_jobs_history = total_jobs_history[len(total_jobs_history)-5:]
            for job_history in total_jobs_history:
                u2exp.write(user_id + " " + str(job_history) + " " + str(counter) + "\n")
                counter += 1
        else:
            for job_history in total_jobs_history:
                u2exp.write(user_id + " " + str(job_history) + " " + str(counter) + "\n")
                counter += 1

        print(user_skills)
        counter = 1
        for skill in user_skills:
            u2skills.write(user_id + " " + str(skill) + " " + str(counter) + "\n")
            counter += 1

        print(user_education)
        counter = 1
        for education in user_education:
            u2edu.write(user_id + " " + str(education) + " " + str(counter) + "\n")
            counter += 1

        print(past_position_ids)
        counter = 1
        if len(past_position_ids) > 4:
            past_position_ids = past_position_ids[len(past_position_ids) -5:]
            for past_position in past_position_ids:
                u2pos_history.write(user_id + " " + str(past_position) + " " + str(counter) + "\n")
                counter += 1
        else:
            for past_position in past_position_ids:
                u2pos_history.write(user_id + " " + str(past_position) + " " + str(counter) + "\n")
                counter += 1

        print(past_companies)
        counter = 1
        if len(past_companies) > 4:
            past_companies = past_companies[len(past_companies) - 5:]
            for past_company in past_companies:
                u2company.write(user_id + " " + str(past_company) + " " + str(counter) + "\n")
                counter += 1
        else:
            for past_company in past_companies:
                u2company.write(user_id + " " + str(past_company) + " " + str(counter) + "\n")
                counter += 1


        # u2exp_Str.write(user_id + " " + current_job + " " + past_companies + " " + str(1) + "\n")

    else:
        print(user_id)
        # u2pos.write(str(k) + "`t" + user_id + "`t" + user_position_id + "`t" + str(0) + "\n")
        # u2exp.write(str(k) + "`t" + user_id + "`t" + total_jobs_history + "`t" + str(0) + "\n")
        # u2skills.write(str(k) + "`t" + user_id + "`t" + user_skills + "`t" + str(0) + "\n")
        # u2edu.write(str(k) + "`t" + user_id + "`t" + user_education + "`t" + str(0) + "\n")
        # u2pos_history.write(str(k) + "`t" + user_id + "`t" + past_position_ids + "`t" + str(0) + "\n")
        # u2company.write(str(k) + "`t" + user_id + "`t" + past_companies + "`t" + str(0) + "\n")

        print("\n\n", type(total_jobs_history))
        print(type(user_skills))
        print(type(user_education))
        print(type(past_position_ids))
        print(type(past_companies))

        print(total_jobs_history)
        print(user_skills)
        print(user_education)
        print(past_position_ids)
        print(past_companies)


        u2pos.write(user_id + " " + user_position_id + " " + str(1) + "\n")

        if isinstance(total_jobs_history, str):
            total_jobs_history = ast.literal_eval(total_jobs_history)

        if isinstance(user_skills, str):
            user_skills = ast.literal_eval(user_skills)

        if isinstance(user_education, str):
            user_education = ast.literal_eval(user_education)

        if isinstance(past_position_ids, str):
            past_position_ids = ast.literal_eval(past_position_ids)

        if isinstance(past_companies, str):
            past_companies = ast.literal_eval(past_companies)

        total_jobs_history = ast.literal_eval(total_jobs_history)
        user_skills = ast.literal_eval(user_skills)
        user_education = ast.literal_eval(user_education)
        past_position_ids = ast.literal_eval(past_position_ids)
        past_companies = ast.literal_eval(past_companies)

        print(total_jobs_history)
        counter = 1
        for job_history in total_jobs_history:
            u2exp.write(user_id + " " + str(job_history) + " " + str(counter) + "\n")
            counter += 1

        print(user_skills)
        counter = 1
        for skill in user_skills:
            u2skills.write(user_id + " " + str(skill) + " " + str(counter) + "\n")
            counter += 1

        print(user_education)
        counter = 1
        for education in user_education:
            u2edu.write(user_id + " " + str(education) + " " + str(counter) + "\n")
            counter += 1

        print(past_position_ids)
        counter = 1
        for past_position in past_position_ids:
            u2pos_history.write(user_id + " " + str(past_position) + " " + str(counter) + "\n")
            counter += 1

        print(past_companies)
        counter = 1
        for past_company in past_companies:
            u2company.write(user_id + " " + str(past_company) + " " + str(counter) + "\n")
            counter += 1

        # u2exp_Str.write(user_id + " " + current_job + " " + past_companies + " " + str(1) + "\n")
        # counter += 1

u2exp.close()
u2skills.close()
u2edu.close()
u2pos.close()
u2pos_history.close()
u2company.close()
