import os
import re
from openai import OpenAI
from tqdm import tqdm
import pandas as pd
from tqdm import tqdm

###############filtering loctions and persons#############

def get_data_from_db(file_path: str) -> pd.Series:
    """retrieves a panda Series of names from a csv file containing notable persons

    Args:
        file_path (str): location of the csv file

    Returns:
        pd.Series: names of notable persons
    """
    if not os.path.exists(file_path):
        return []
    df = pd.read_csv(file_path, encoding='utf-8')
    df.dropna()
    return df.iloc[:, 0]

def contains_location(caption: str):
    if isinstance(caption, float):
        return False
    caption_words = caption.split()
    for location in LOCATION_DB:
        if isinstance(location, str):
            location_words = location.split()
            if all(word in caption_words for word in location_words):
                return True
    return False

def contains_person_name(caption: str):
    if isinstance(caption, float):
        return False
    caption_words = caption.split()
    for name in PERSON_DB:
        name_words = name.split()
        for subname in name_words:
            if subname in caption_words:
                return True
    return False

def remove_people_and_locations_from_concepts(concepts: pd.DataFrame) -> pd.DataFrame:
    """takes in a file and removes all persons and locations from the column "concepts"

    Args:
        concepts (pd.DataFrame): The dataframe on which notable persons location should be filtere
        name_db_file_path (str): path of the notable persons database. Defaults to 

    Returns:
        pd.DataFrame: filtered concepts without notable persons and locations
    """
    
    condition = concepts['concepts'].apply(lambda concept: not contains_person_name(concept))
    filtered_df = concepts[condition]

    return filtered_df

##############GPT filtering##############################
# set OpenAi_key in shell to execute the code underneath
client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY"),
)

def create_empty_csv(filepath: str):
    columns = ["concept", "country", "semantic_field", "language"]
    
    if not os.path.exists(filepath):
        # Create an empty DataFrame with the specified columns
        df = pd.DataFrame(columns=columns)
        
        # Save the DataFrame to a CSV file
        df.to_csv(filepath, index=False)

def get_single_response(concept):
    message = "You are a Cultural Expert who knows all the answers to culturally specific questions. Answer the following questions.\
    Question: What country is the following concept from? Choose from the countries:  China, Germany, Spain, South Korea, None.\
    Which category does the concept belong to? Choose the category from the concepts: Beverages, Celebration, Clothing, Food, Fruit, Houses, Music, Religion, Sport, Utensil, Vegetable, Visual Arts, Food, None. \
    In which language is the concept written? Choose the language from Chinese, English, German, Korean, Spanish, None.\
    Provide the answer in the form: {Concept: Horchata, Country: Spain, Category: Beverages, Language: Spanish}.\
    Concept: " + concept+ "Your Answer:\
    When providing the answer, answer in the requested format."

    completion = client.chat.completions.create(model = "gpt-3.5-turbo",
                                                messages=[{"role": "user", "content": message}], temperature=0.5, max_tokens=600)
    
    return completion.choices[0].message.content

def evaluate_concepts_GPT(concepts: pd.DataFrame, gold: str, candidates:str, drop:str, unassigned: str):
    """evaluates the concepts and assigns them into 4 different dataframes depending on how well they match our chosen cultures and semantic fields 

    Args:
        concepts (pd.DataFrame): the concepts to evaluate
        gold (str): the concepts that have no none in their answer generation
        candidates (str): concepts for which none appears at least one but less than three times
        drop (str): dropped concepts for which multiple categories are none
        unassigned (str): concepts that do not match our expression matching 
    """

    create_empty_csv(drop)
    create_empty_csv(gold)
    create_empty_csv(candidates)
    
    if not os.path.exists(unassigned):
        with open(unassigned, 'w') as file:
            pass  
    
    # apply the GPT prompt to every concept candidate
    for concept in tqdm(concepts.iloc[:, 0]):
        answer = get_single_response(concept)
        # captures the relevant answer components of the model answer i.e. Country:Germany, Category:Beverages 
        pattern = r':\s(.*?)(?=,|$)'
        matches = re.findall(pattern, answer)
        # answers with too little information are saved in a seperate file
        if len(matches) < 4:
            with open(unassigned, 'a') as file:
                file.write(concept + ", ".join(matches) + "\n")
            continue
    
        new_row = pd.DataFrame([{
            "concept": concept,
            "country": matches[1],
            "semantic_field": matches[2],
            "language": matches[3]
        }])

    	# sort the concept candidates in our three categories, depending on how often none appears in the model response
        lowercase_list = [element.lower() for element in matches]
        count = lowercase_list.count("none")
        if count == 0:
            new_row.to_csv(gold, mode='a', header=not pd.io.common.file_exists(gold), index=False)
        elif count == 3:
            new_row.to_csv(drop, mode='a', header=not pd.io.common.file_exists(drop), index=False)
        else:
            new_row.to_csv(candidates, mode='a', header=not pd.io.common.file_exists(candidates), index=False)

def cleanup_format(old_file: str, new_file:str):

    # remove {} brackets at the beginning or end of the row of the concept candidates
    with open(old_file, mode="r", encoding="utf-8") as f:
        with open(new_file, 'a', encoding="utf-8") as file:
            for line in f.readlines():
                line = line.lower()
                if line[0] == "{":
                    file.write(line[10:-2] + "\n")
                else:
                    file.write(line[:-2] + "\n")

def sort_csv(file_name: str):
    df = pd.read_csv(file_name)

    # Strip whitespace from all string columns
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # lowercase all names
    df = df.applymap(lambda x: x.lower() if isinstance(x, str) else str(x))

    # Sort the DataFrame first by the second column, then by the third column
    sorted_df = df.sort_values(by=[df.columns[1], df.columns[2], df.columns[0]])

    # Save the sorted DataFrame to the same CSV file
    sorted_df.to_csv(file_name[:-4] + "_sorted" +".csv", index=False, encoding='utf-8')

def seperate_incorrect_gpt_assignments(concepts: pd.DataFrame):
    """seperates the concepts further into gold_concept candidates and more general concepts based on the language 

    Args:
        concepts (pd.DataFrame): the concepts to further seperate
    """
    gold_concepts = []
    general_concepts = []
    cultures = ["germany", "south korea", "china", "spain"]
    semantic_fields = ["beverages", "celebration", "food", "clothing", "fruit", "houses", "music", "religion", "sport", "utensil", "vegetable", "visual arts"]
    
    concepts = concepts.applymap(lambda x: x.lower() if isinstance(x, str) else str(x))

    # ensure to only include concepts that are in the language of one our target cultures or in english
    for index, row in concepts.iterrows():

        if row["country"] in cultures and row["semantic_field"] in semantic_fields:
            if row["language"] in "german":
                row["language"] = "german"
                gold_concepts.append(row.to_dict())

            elif row["language"] in "korean":
                row["language"] = "korean"
                gold_concepts.append(row.to_dict())

            elif row["language"] in "chinese":
                row["language"] = "chinese"
                gold_concepts.append(row.to_dict())

            elif row["language"] in "spanish":
                row["language"] = "spanish"
                gold_concepts.append(row.to_dict())
            elif row["language"] in "english":
                row["language"] = "english"
                gold_concepts.append(row.to_dict())
            else: 
                general_concepts.append(row.to_dict())
        else:
            general_concepts.append(row.to_dict())

    gold = pd.DataFrame(gold_concepts, columns=concepts.columns)
    general = pd.DataFrame(general_concepts, columns=concepts.columns)
    gold.to_csv("output/gold_candidates.csv", encoding='utf-8', index=False)
    general.to_csv("output/general.csv", encoding='utf-8', index=False)

def seperate_preassigned_concepts(concepts: pd.DataFrame):
    """Seperates the concepts into candidates with a preassigned culture and semantic field

    Args:
        concepts (pd.DataFrame): the concepts to seperate
    """
    # perform an additional assessment to only include concept candidates that do have a culture and semantic field assignment 
    double_preassigned_concepts = []
    single_preassigned_concepts = []
    noisy_concepts = []
    for index, row in concepts.iterrows():
        if pd.notna(row.iloc[1]) and row.iloc[1].strip() == "indian":
            continue
        elif pd.notna(row.iloc[1]) and row.iloc[1].strip() != "" and pd.notna(row.iloc[2]) and row.iloc[2].strip() != "":
            double_preassigned_concepts.append(row.to_dict())
        elif pd.notna(row.iloc[1]) and row.iloc[1].strip() != "" or pd.notna(row.iloc[2]) and row.iloc[2].strip() != "":
            single_preassigned_concepts.append(row.to_dict())
        else:
            noisy_concepts.append(row.to_dict())

    single_assign_concepts = pd.DataFrame(single_preassigned_concepts, columns=concepts.columns)
    gold_eval_concepts = pd.DataFrame(double_preassigned_concepts, columns=concepts.columns)
    noisy_df = pd.DataFrame(noisy_concepts, columns=concepts.columns)
    single_assign_concepts.to_csv("output/candidates.csv", encoding='utf-8', index=False)
    gold_eval_concepts.to_csv("output/gold_eval.csv", encoding='utf-8', index=False)
    noisy_df.to_csv("output/noisy_concepts.csv",  encoding='utf-8', index=False)

def convert_to_xlsx(input_file: str, output_file:str):

    df = pd.read_csv(input_file)
    df_sorted = df.sort_values(by=['country', 'semantic_field', 'language'])
    df_sorted.to_excel(output_file, index=False)

LOCATION_DB = get_data_from_db("data/GeoNames_DB.csv")
PERSON_DB = get_data_from_db("data/name_db.csv")
CONCEPT_FILE = "output/concept_candidates.csv"

# remove duplicate concepts
df_concepts = pd.read_csv(CONCEPT_FILE)
duplicates = df_concepts["concepts"].duplicated()
df_cleaned = df_concepts[~duplicates]

# filter out persons and locations 
filtered_concepts = remove_people_and_locations_from_concepts(df_cleaned)
filtered_concepts.to_csv("output/concepts_filtered.csv", index=False)

# gpt evalution
evaluate_concepts_GPT(filtered_concepts, "output/gpt_gold_candidates.csv", "output/gpt_concept_candidates.csv", "output/gpt_dropped_concepts.csv", "output/gpt_unassigned_concepts.csv")
df = pd.read_csv("output/gpt_gold_candidates.csv")
seperate_preassigned_concepts(df)
cleanup_format("output/gold_eval.csv", "output/cleaned_gold_eval.csv")
df_gold = pd.read_csv("output/cleaned_gold_eval.csv")
seperate_incorrect_gpt_assignments(df_gold)


convert_to_xlsx("output/gold_candidates.csv", "manual_assessment_concepts.xlsx")
