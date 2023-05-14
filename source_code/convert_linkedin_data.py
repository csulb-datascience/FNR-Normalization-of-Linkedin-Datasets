import re
import unicodedata
import os
import pandas as pd
import ast
from collections import defaultdict
import csv
import string

def normalize_job_title(title):
    title = unicodedata.normalize('NFKC', title)

    if re.match('co-founder', title, re.IGNORECASE):
        title = re.sub('co-founder', 'Founder', title, flags=re.IGNORECASE)
        return title.rstrip()

    if re.match('(?=.*r&d)(?=.*R&D)', title, re.IGNORECASE):
        title = re.sub('r&d', 'Research', title, flags=re.IGNORECASE)
        return title.rstrip()

    # Strip values with interns

    # if re.match('(.*\sintern)', title, re.IGNORECASE):
    #     title =

    # Strip " Jr or jr or jr. to Junior"

    if re.match('jr', title, re.IGNORECASE):
        title = title.replace('.', '')
        title = re.sub(r"jr", "Junior", title, flags=re.IGNORECASE)

    # Strip " Sr or sr or sr. to Senior"

    if re.match('sr', title, re.IGNORECASE):
        title = title.replace('.', '')
        title = re.sub(r"sr", "Senior", title, flags=re.IGNORECASE)

    # Strip " SWE or SDE to Software Engineer"

    if re.match('swe', title, re.IGNORECASE) or re.match('sde', title, re.IGNORECASE) or re.match('sdet', title, re.IGNORECASE):
        title = "Software Engineer"

    # Strip " UI or UX to Web Designer"

    if re.match('ui', title, re.IGNORECASE) or re.match('ux', title, re.IGNORECASE):
        title = "Web Designer"

    # Strip " a/ b"
    try:
        title = re.match("(.*)\/", title, re.IGNORECASE).group(1)
    except:
        pass

    # Strip " a/ b"
    try:
        title = re.search('(.*)(?=\/)', title, re.IGNORECASE).group(1)
    except:
        pass

    # Strip " c and d"
    try:
        title = re.search('(.*)\s(?=and)', title, re.IGNORECASE).group(0)
    except:
        pass

    # Strip " c & d"
    try:
        title = re.search('(.*)\s(?=&)', title, re.IGNORECASE).group(0)
    except:
        pass

    # Strip " c, d"
    try:
        title = re.search('(.*)(?=,)', title, re.IGNORECASE).group(0)
    except:
        pass

    # Strip " c in d"
    try:
        title = re.search('(.*)\s(?=\bin\b)', title, re.IGNORECASE).group(0)
    except:
        pass

    # Strip " c: d"
    try:
        title = re.search('(.*)(?=:)', title, re.IGNORECASE).group(0)
    except:
        pass

    # Strip " c - d"
    try:
        if re.match('front-end', title, re.IGNORECASE):
            title = "Front-End Engineer"
        else:
            title = re.search('(.*)\s(?=-)', title, re.IGNORECASE).group(0)
            title = title.split("-")[0]

    except:
        pass

    title = title.replace("+", "")
    title = title.replace("*", "")
    title = re.sub(r'【.*】', '', title)
    title = re.sub(r'\[.*\]', '', title)
    title = re.sub(r'「.*」', '', title)
    title = re.sub(r'\(.*\)', '', title)
    title = re.sub(r'\<.*\>', '', title)
    title = re.sub(r'[※@◎].*$', '', title)
    # title = re.sub(r'-.*', '', title)
    title = re.sub(r'\(.*', '', title)
    title = re.sub(r'(\bat\b)', '', title)
    title = re.sub(r"\b(?=[MDCLXVIΙ])M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})([IΙ]X|[II]V|V?[II]{0,3})\b\.?", '', title)
    title = re.sub(" \|.*", "", title)
    title = title.rstrip().lower()
    title = string.capwords(title)
    return title

def create_updated_linkedin_csv(file):
    df = pd.read_csv(os.getcwd() + '/datasets/Linkedin/linkedin_data.csv')

    tempDict = {}

    for i, row in df.iterrows():
        tempDict[i] = row

    final = {}

    for k, v in tempDict.items():
        final[v[0]] = ast.literal_eval(v[1])

    info = ['id', 'name', 'years of experience', 'current_job', 'current_job_company_id', 'current_job_id',
            'current_company', 'current_company_id', 'education_degree', 'education_degree_type', 'education_ids',
            'skills',
            'skills_ids', 'industries', 'total_jobs_history', 'total_companies_history', 'past_jobs', 'past_jobs_ids',
            'past_companies', 'past_education_degree', 'past_education_type']

    for k, v in final.items():
        print("\nk : ", k)
        print("v : ", v)
        print(v.values())

    with open(os.getcwd() + '/datasets/Linkedin/updated_fixed_linkedin.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(info)

        for k, v in final.items():
           writer.writerow(v.values())

def csv2txt(directory, file, normalize=False):
    file = os.getcwd() + "/datasets/" + directory + "/" + file

    df = pd.read_csv(os.getcwd() + "/datasets/Linkedin/updated_fixed_linkedin.csv")

    df = df.where(pd.notnull(df), None)

    print(os.path.abspath(os.getcwd()))

    tempDict = {}

    for i, row in df.iterrows():
        tempDict[i] = row

    u2skills = open(file + "_u2s.txt", "w", encoding="utf-8")
    u2pos = open(file + "_u2pos.txt", "w", encoding="utf-8")
    u2exp = open(file + "_u2exp.txt", "w", encoding="utf-8")
    u2edu = open(file + "_u2edu.txt", "w", encoding="utf-8")
    u2pos_history = open(file + "_u2posHistory.txt", "w", encoding="utf-8")
    u2company = open(file + "_u2company.txt", "w", encoding="utf-8")
    u2skill_Str = open(file + "_u2skillStr.txt", "w", encoding="utf-8")
    u2exp_Str = open(file + "_u2expStr.txt", "w", encoding="utf-8")
    u2past_job_Str = open(file + "_u2past_job_Str.txt", "w", encoding="utf-8")

    counter = 0

    if normalize:
        for k, v in tempDict.items():
            user_id = str(v["id"])
            user_name = str(v["name"])
            user_position_id = str(v["current_job_id"])
            user_current_company = str(v["current_company_id"])
            user_current_job_id = str(v["current_job_company_id"])
            user_jobs = str(v["total_jobs_history"])
            user_past_jobs = str(v["past_jobs"])
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
                u2pos.write(user_id + " " + user_position_id + " " + str(1) + "\n")

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

                try:
                    if isinstance(user_past_jobs, str):
                        user_past_jobs = list(ast.literal_eval(user_past_jobs))
                except:
                    pass

                counter = 1
                if len(total_jobs_history) > 4:
                    total_jobs_history = total_jobs_history[len(total_jobs_history) - 5:]
                    for job_history in total_jobs_history:
                        u2exp.write(user_id + " " + str(job_history) + " " + str(counter) + "\n")
                        counter += 1
                else:
                    for job_history in total_jobs_history:
                        u2exp.write(user_id + " " + str(job_history) + " " + str(counter) + "\n")
                        counter += 1

                counter = 1
                if len(user_past_jobs) >= 4:
                    user_past_jobs = user_past_jobs[len(user_past_jobs) - 3:]
                    u2past_job_Str.write(user_id + " " + normalize_job_title(str(current_job)) + " " + str(counter) + "\n")
                    for past_job in user_past_jobs:
                        counter += 1
                        u2past_job_Str.write(user_id + " " + normalize_job_title(str(past_job)) + " " + str(counter) + "\n")
                else:
                    u2past_job_Str.write(
                        user_id + " " + normalize_job_title(str(current_job)) + " " + str(counter) + "\n")
                    for past_job in user_past_jobs:
                        counter += 1
                        u2past_job_Str.write(
                            user_id + " " + normalize_job_title(str(past_job)) + " " + str(counter) + "\n")

                counter = 1
                for skill in user_skills:
                    u2skills.write(user_id + " " + str(skill) + " " + str(counter) + "\n")
                    counter += 1

                counter = 1
                for education in user_education:
                    u2edu.write(user_id + " " + str(education) + " " + str(counter) + "\n")
                    counter += 1

                counter = 1
                if len(past_position_ids) > 4:
                    past_position_ids = past_position_ids[len(past_position_ids) - 5:]
                    for past_position in past_position_ids:
                        u2pos_history.write(user_id + " " + str(past_position) + " " + str(counter) + "\n")
                        counter += 1
                else:
                    for past_position in past_position_ids:
                        u2pos_history.write(user_id + " " + str(past_position) + " " + str(counter) + "\n")
                        counter += 1

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

                u2exp_Str.write(user_id + " " + normalize_job_title(str(current_job)) + " " + str(past_companies) + " " + str(1) + "\n")
            else:
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



                u2exp_Str.write(user_id + " " + normalize_job_title(str(current_job)) + " " + str(past_companies) + " " + str(0) + "\n")
                # counter += 1
    else:
        for k, v in tempDict.items():
            user_id = str(v["id"])
            user_name = str(v["name"])
            user_position_id = str(v["current_job_id"])
            user_current_company = str(v["current_company_id"])
            user_current_job_id = str(v["current_job_company_id"])
            user_jobs = str(v["total_jobs_history"])
            user_past_jobs = str(v["past_jobs"])
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
                u2pos.write(user_id + " " + user_position_id + " " + str(1) + "\n")

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

                counter = 1
                if len(total_jobs_history) > 4:
                    total_jobs_history = total_jobs_history[len(total_jobs_history) - 5:]
                    for job_history in total_jobs_history:
                        u2exp.write(user_id + " " + str(job_history) + " " + str(counter) + "\n")
                        counter += 1
                else:
                    for job_history in total_jobs_history:
                        u2exp.write(user_id + " " + str(job_history) + " " + str(counter) + "\n")
                        counter += 1

                counter = 1
                if len(user_past_jobs) > 4:
                    user_past_jobs = user_past_jobs[len(user_past_jobs) - 3:]
                    u2past_job_Str.write(
                        user_id + " " + normalize_job_title(str(current_job)) + " " + str(counter) + "\n")
                    for past_job in user_past_jobs:
                        counter += 1
                        u2past_job_Str.write(
                            user_id + " " + normalize_job_title(str(past_job)) + " " + str(counter) + "\n")
                else:
                    u2past_job_Str.write(
                        user_id + " " + normalize_job_title(str(current_job)) + " " + str(counter) + "\n")
                    for past_job in user_past_jobs:
                        counter += 1
                        u2past_job_Str.write(
                            user_id + " " + normalize_job_title(str(past_job)) + " " + str(counter) + "\n")


                counter = 1
                for skill in user_skills:
                    u2skills.write(user_id + " " + str(skill) + " " + str(counter) + "\n")
                    counter += 1

                counter = 1
                for education in user_education:
                    u2edu.write(user_id + " " + str(education) + " " + str(counter) + "\n")
                    counter += 1

                counter = 1
                if len(past_position_ids) > 4:
                    past_position_ids = past_position_ids[len(past_position_ids) - 5:]
                    for past_position in past_position_ids:
                        u2pos_history.write(user_id + " " + str(past_position) + " " + str(counter) + "\n")
                        counter += 1
                else:
                    for past_position in past_position_ids:
                        u2pos_history.write(user_id + " " + str(past_position) + " " + str(counter) + "\n")
                        counter += 1

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

                u2exp_Str.write(user_id + " " + str(current_job) + " " + str(past_companies) + " " + str(1) + "\n")
            else:
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

                print(user_past_jobs)

                u2exp_Str.write(user_id + " " + str(current_job) + " " + str(past_companies) + " " + str(0) + "\n")
                # counter += 1

    u2exp.close()
    u2skills.close()
    u2edu.close()
    u2pos.close()
    u2pos_history.close()
    u2company.close()


csv2txt(directory="Linkedin", file="updated_fixed_linkedin", normalize=True)

# print(normalize_job_title("Front-End Engineer"))