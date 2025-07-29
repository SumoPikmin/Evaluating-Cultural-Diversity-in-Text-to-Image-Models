
##############################################
import os
import random
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup, Tag
from typing import Tuple
import requests


############## helper functions file manipulation##############

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

def overwrite_file(filename:str, new_contents: list[str] | pd.DataFrame, file_type="csv") -> None:
    """overwrites an existing file by adding new contents to the file and removing duplicates.
       if the file does not exist, a new file is created instead in which the new contents are written in

    Args:
        filename (str): The name of the file to overwrite
        new_contents (list[str] | pd.DataFrame): The data that will be added to the file. The type depends on the file type 
        file_type (str, optional): The type of the file. Defaults to "csv".

    Raises:
        Exception: If a datatype other than a pandas Dataframe is used for a .csv file
        Exception: If an invalid file type is given, other than csv
    """
    if file_type == "csv":
        if not isinstance(new_contents, pd.DataFrame):
            raise Exception("Use a panda dataframe for writing into a csv file")
        file_path = filename
        old_contents = pd.read_csv(file_path)
        if old_contents is None:
            new_contents.to_csv(file_path, index=False)
        else:
            merged_contents = concat_and_filter_unique_values(old_contents, new_contents)
            merged_contents.to_csv(file_path, index=False)
    else: 
        raise Exception("invalid file_type, only csvs are accepted")

############## helper functions for extraction and structuring##############
def concat_and_filter_unique_values(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """concatenates two dataframes and removes all multiple appearances of duplicate values

    Args:
        df1 (pd.DataFrame): The first dataframe
        df2 (pd.DataFrame): The second dataframe

    Returns:
        pd.DataFrame: The concatenation of both dataframes with only unique values 
    """
    combined_df = pd.concat([df1, df2])
    return combined_df.drop_duplicates(subset=df1.columns)

def merge_csvs(file1, file2):

    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Concatenate the two dataframes
    merged_df = pd.concat([df1, df2])

    # Drop duplicate rows
    merged_df.drop_duplicates(inplace=True)

    # Save the merged dataframe to a new CSV file
    merged_df.to_csv('merged_file.csv', index=False)

def find_paragraphs_for_concepts(soup: Tag, concepts: pd.DataFrame)-> pd.DataFrame:
    """Finds the paragraph the concepts appear in and writes it in an additional "information" column

    Args:
        soup (Tag): beautiful soup tag
        concepts (pd.DataFrame): concepts to find the paragraphs for

    Returns:
        pd.DataFrame: the original concepts with an additional column information
    """
    all_paragraphs = soup.select("p")
    new_concepts = concepts.copy()
    new_concepts["information"] =""
    for idx, concept in new_concepts.iterrows():
        for para in all_paragraphs:
            para_text = para.get_text().lower()
            if concept["concepts"] in para_text:
                if concept["information"] == "":
                    new_concepts.at[idx, "information"] = para_text
                else:
                    new_concepts.at[idx, "information"] += ", " + para_text
    return new_concepts

def get_all_italic_words(soup_content: Tag) -> pd.DataFrame:
    """The bodycontent of the wikipedia Page to retrieve the data from

    Args:
        soup_content (Tag): The bodycontent of the wikipedia Page to retrieve the data from

    Returns:
        pd.DataFrame: a dataframe containing all words written in italic with the head: "concepts"
    """
    italic_words = set()
    italic_tags = soup_content.find_all("i")
    for tag in italic_tags:
        italic_phrase = tag.text.strip()
        if italic_phrase.isalpha() and len(italic_phrase) > 1:
            italic_words.add(italic_phrase.lower())
    return pd.DataFrame({"concepts": list(italic_words)})

def contains_person_name(caption: str):
    if isinstance(caption, float):
        return False
    caption_words = caption.split()
    for name in PERSON_DB:
        name_words = name.split()
        if all(word in caption_words for word in name_words):
            return True
    return False

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

def get_captioned_images(soup_content: Tag) -> pd.DataFrame:
    """Returns all image source urls of the wikipedia page along their captions or image titles

    Args:
        soup_content (Tag): The bodycontent of the wikipedia Page to retrieve the data from

    Returns:
        pd.DataFrame: Returns image urls and their captions with the columns "img_source" and "img_captions"
    """
    img_captions = []
    img_sources = []
 
    for fig in soup_content.find_all("figure"):
        img = fig.find("img")

        if img:
            img_caption = fig.find("figcaption").get_text().strip().lower()
            img_src = img.get("src")
            if any(char.isdigit() for char in img_caption) or not contains_person_name(img_caption):
                if img_src.endswith(("jpg", "jpeg", "gif")):
                    img_captions.append(img_caption)
                    img_sources.append(img.get("src")) 

    for img in soup_content.find_all("img"):
        img_src = img.get("src") 
        img_title = img.get("alt")

        if (img_src not in img_sources and
            img_src.endswith(("jpg", "jpeg", "gif")) and
            img_title and
            img_title != ""):
                img_title = img_title.strip().lower()
                if any(char.isdigit() for char in img_title) or not contains_person_name(img_title):
                    img_captions.append(img_title)
                    img_sources.append(img_src)

    return pd.DataFrame({'img_source': img_sources,'img_captions': img_captions})

def extract_link_concepts_and_urls(soup_content: Tag, current_url: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns all extracted urls leading to other wikipedia pages of the wikipedia page and
       concepts extracted from the urls.

    Args:
        soup_content (Tag): The bodycontent of the wikipedia Page to retrieve the data from

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Urls and url concepts of the webpage with the columns "url" and "concepts" 
    """
    url_concepts = set()
    scraped_urls = set()
    
    language_prefixes = {"en": "References", "es":"Referencias", "de":"Weblinks", "zh": "參考來源", "ko":"참고 문헌"}
    language_code = current_url.split('//')[1][:2]
    if language_code in language_prefixes:
        id = language_prefixes[language_code]
    else:
        return pd.DataFrame({"urls":[]}), pd.DataFrame({"concepts": []})
    
    reference_section = soup_content.find(id=id)
    outgoing_links = soup_content.find_all("a")
    for link in outgoing_links:
        concept_title = link.get("title")
        href = link.get("href")
        if reference_section and href == reference_section.find_next("a").get("href"):
            break
        if href and href.startswith("/wiki/") and ':' not in href and "(identifier)" not in href:
            if concept_title.isalpha() and len(concept_title) > 1:
                url_concepts.add(concept_title.lower())
            scraped_urls.add(href)
    return pd.DataFrame({"urls":list(scraped_urls)}), pd.DataFrame({"concepts": list(url_concepts)})

def get_default_culture_and_semantic_field(title: str) -> Tuple[str, str]:
    """returns culture and semantic field for preassignment by using keywords based on the title

    Args:
        title (str): title of the web page

    Returns:
        Tuple[str, str]: preassigned culture and semantic field of the title
    """
    culture = ""
    semantic_field = ""
    for keyword in CULTURES.keys():
        if keyword in title:
            culture = CULTURES[keyword]
            break

    for keyword in SEMANTIC_FIELDS.keys():
        if keyword in title:
            semantic_field = SEMANTIC_FIELDS[keyword]
            break

    return culture, semantic_field

def assign_culture_and_semantic_fields_to_concepts(concepts: pd.DataFrame) -> pd.DataFrame:
    """Assigns cultures to a dataframe of concepts by analyzing title and summary of the wikipedia page where the concepts stem from.

    Args:
        title (str): title of the wikipedia page
        summary (str): intro text of the wikipedia page
        cultures (_type_): a dictionary mapping from keywords to cultures
        concepts (pd.DataFrame): A dataframe of the concepts which cultures will be assigned to

    Returns:
        pd.DataFrame: Dataframe with the concepts and an added column "culture" that assigns each concept a culture
    """
    for idx, concept in concepts.iterrows():
        for keyword in CULTURES.keys():
            if keyword in concept["information"]:
                concepts.at[idx, "culture"] = CULTURES[keyword]
                break
            
    for idx, concept in concepts.iterrows():
        for keyword in SEMANTIC_FIELDS.keys():
            if keyword in concept["information"]:
                concepts.at[idx, "semantic_field"] = SEMANTIC_FIELDS[keyword]
                break

    return concepts.drop('information', axis=1)

def remove_people_and_locations_from_concepts(concepts: pd.DataFrame) -> pd.DataFrame:
    """takes in a file and removes all persons and locations from the column "concepts"

    Args:
        concepts (pd.DataFrame): The dataframe on which notable persons location should be filtere
        name_db_file_path (str): path of the notable persons database. Defaults to 

    Returns:
        pd.DataFrame: filtered concepts without notable persons and locations
    """
    
    # condition = concepts['concepts'].apply(lambda concept: not contains_person_name(concept) and not contains_location(concept))
    condition = concepts['concepts'].apply(lambda concept: not contains_location(concept))
    filtered_df = concepts[condition]

    return filtered_df

def append_multilingual_url(default_culture: str, next_urls: list[str], visited_urls: list[str]) -> None:
    """appends the multilingual version of a web page if it exists based on its langauge code

    Args:
        default_culture (str): preassigned culture of the web page
        next_urls (list[str]): list of the next urls to visit
        visited_urls (list[str]): list of the already visited urls
    """
    language_prefixes = {"korean": "ko", "chinese":"zh", "spanish":"es", "german":"de"}
    lang_dict = {}
    for lang_pairs in soup.select("li.interlanguage-link > a"):
        lang_dict[lang_pairs.get('lang')] = lang_pairs.get('href')

    if default_culture in language_prefixes and language_prefixes[default_culture] in lang_dict:
        new_link = lang_dict[language_prefixes[default_culture]]
        if  new_link not in visited_urls:
            next_urls.append(new_link)

def select_next_urls_to_jump_to():
    """selects the next urls to visit

    Returns:
        next_urls list[string]: list of the urls to visit next 
    """
    url_df = pd.read_csv_file(ALL_URLS_FILE)
    # select 100 urls to append and only append urls not visited yet
    new_urls = url_df["urls"].iloc[:100]
    
    if visited_urls:
        next_urls = ["https://en.wikipedia.org/" + str(url) for url in new_urls if "https://en.wikipedia.org/" + str(url) not in visited_urls]
    else: 
        next_urls = ["https://en.wikipedia.org/" + str(url) if not url.startswith("https://") else str(url) for url in new_urls ]
    # remove the 100 urls from the queue
    url_df["urls"].iloc[100:].to_csv(ALL_URLS_FILE, index=False)
    return next_urls


LOCATION_DB = get_data_from_db("data/GeoNames_DB.csv")
PERSON_DB = get_data_from_db("data/name_db.csv")
START_URLS_FILE = "data/start_urls.csv"


CULTURES = {"german":"german", "germany":"german", "spanish":"spanish", "spain":"spanish", "chinese":"chinese", "china": "chinese", "korean": "korean", "korea": "korean",
            "deutsch":"german", "deutschland": "german", "spanisch":"spanish", "spanien":"spanish", "chinesisch":"chinese", "china": "chinese", "koreanisch": "korean", "korea":"korean",
            "alemán":"german", "alemania": "german", "español":"spanish", "españa":"spanish", "chino":"chinese", "china": "chinese", "coreano": "korean", 
            "德语":"german", "德国": "german", "西班牙":"spanish", "中文":"chinese", "中国": "chinese", "汉语": "chinese", "韩语": "korean", "韩国": "korean",
            "andalusia": "spanish", "aragon": "spanish", "asturias": "spanish", "balearic islands":"spanish", "basque country": "spanish", "canary islands": "spanish", "cantabria": "spanish", "castile and león": "spanish", "castilla-la mancha": "spanish", "catalonia": "spanish", "extremadura": "spanish", "galicia": "spanish", "la rioja": "spanish", "madrid": "spanish", "murcia": "spanish", "navarre": "spanish",
            "baden-württemberg": "german", "bavaria": "german", "berlin": "german", "brandenburg": "german", "bremen": "german", "hamburg": "german", "hesse": "german", "lower saxony": "german", "mecklenburg-vorpommern": "german", "north rhine-westphalia": "german", "rhineland-palatinate": "german", "saarland":"german", "saxony":"german", "saxony-anhalt":"german", "schleswig-holstein":"german", "thuringia": "german",
            "hebei": "chinese", "shanxi": "chinese", "liaoning": "chinese", "jilin": "chinese", "heilongjiang": "chinese", "jiangsu":"chinese", "zhejiang": "chinese", "anhui": "chinese", "fujian": "chinese", "jiangxi": "chinese", "shangdong": "chinese", "henan": "chinese", "hunan": "chinese", "guangdong": "chinese", "hainan": "chinese", "sichuan": "chinese", "guizhuo": "chinese", "yunnan": "chinese", "shanxi": "chinese", "gansu": "chinese", "qinghai":"chinese",
            "chungcheong": "korean", "gangwon":"korean", "gyeonggi": "korean", "gyeongsang": "korean", "jeonbuk": "korean", "jeolla":"korean", "jeju":"korean"
            }

SEMANTIC_FIELDS = {"beverage":"beverages", "alcohol": "beverages", "beer": "beverages", "wine":"beverages", "celebration":"celebration","dish": "food", "cuisine": "food", "food": "food", "clothing": "clothing", "fruit":"fruit", "houses": "houses", "instrument":"music", "religion": "religion", "sport": "sport","utensil": "utensil", "tool": "utensil", "vegetable":"vegetable", "visual arts": "visual arts",
                   "getränk":"beverages", "feier":"celebration", "essen": "food", "kleidung": "clothing", "obst":"fruit","haus":"houses", "häuser": "houses", "werkzeug": "utensil", "gemüse":"vegetable", "bildende kunst": "visual arts",
                    "bebida":"beverages", "celebración":"celebration","cocina": "food", "comida": "food", "ropa": "clothing", "fruta":"fruit", "casa": "houses", "instrumento":"music", "religión": "religion", "deporte": "sport","utensilio": "utensil", "herramienta": "utensil", "verdura":"vegetable", "artes visuales": "visual arts",
                    "饮料":"beverages", "庆祝":"celebration", "食物": "food", "衣服": "clothing", "水果":"fruit", "房子": "houses", "乐器":"music", "宗教": "religion", "运动": "sport","器具": "utensil", "工具": "utensil", "蔬菜":"vegetable", "视觉艺术": "visual arts",
                    "음료": "beverages", "축하 ": "celebration", "축하 ": "food", "옷": "clothing", "과일": "fruit", "집": "houses", "악기": "music", "종교": "religion", "스포츠": "sport", "도구": "utensil", "도구": "utensil", "채소": "vegetable", "시각 예술": "visual arts"
                    }
VISITED_URLS_FILE = "output/visited_urls.csv"
ALL_URLS_FILE = "output/all_urls.csv"
CONCEPT_FILE = "output/concept_candidates.csv"
IMAGES_FILE = "output/images.csv"

ITERATIONS = 15000

df_next_urls_ = pd.read_csv(START_URLS_FILE)
next_urls = df_next_urls_["urls"].head(200).tolist()
random.shuffle(next_urls)
empty_df = pd.DataFrame(columns=df_next_urls_.columns)
empty_df.to_csv(ALL_URLS_FILE, index=False)
visited_urls =[]

for i in tqdm(range(ITERATIONS)):
    response = requests.get(url=next_urls[0])
    visited_urls.append(next_urls[0])

    if response:
        soup = BeautifulSoup(response.content, 'html.parser')
        soup_content = soup.find(id="bodyContent")

        title = soup.select("#firstHeading")[0].text.lower()
        default_culture, default_semantic_field = get_default_culture_and_semantic_field(title)
        append_multilingual_url(default_culture, next_urls, visited_urls)

        # extract images from the web page
        page_images = get_captioned_images(soup_content)
        page_images["culture"] = default_culture
        page_images["semantic_field"] = default_semantic_field
        overwrite_file(IMAGES_FILE, page_images)
        
        # extract urls and add the urls that have not been visited yet
        outgoing_links, url_concepts = extract_link_concepts_and_urls(soup_content, next_urls[0])
        page_outgoing_urls = outgoing_links[~outgoing_links["urls"].isin(visited_urls)]
        overwrite_file(ALL_URLS_FILE, page_outgoing_urls)
        

        # extract concepts from links and italic words
        concepts = concat_and_filter_unique_values(url_concepts, get_all_italic_words(soup_content))
        concepts["culture"] = default_culture
        concepts["semantic_field"] = default_semantic_field
        concepts_with_information = find_paragraphs_for_concepts(soup, concepts)
        concepts_culture = assign_culture_and_semantic_fields_to_concepts(concepts_with_information)

        overwrite_file(CONCEPT_FILE, concepts_culture)

    next_urls = next_urls[1:]
    # parse new urls if there are no urls to jumps to
    if not next_urls:
        next_urls = select_next_urls_to_jump_to()
    if i % 100 == 0:
        tqdm.write(f"Processed {i} iterations")

