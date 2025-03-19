import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

URL = "https://www.vogue.com/fashion/trends"

def scrape_fashion_trends():
    response = requests.get(URL)
    soup = BeautifulSoup(response.text, "html.parser")

    articles = soup.find_all("div", class_="article-content")[:20]  
    data = []

    for article in articles:
        title = article.find("h2").text if article.find("h2") else "No Title"
        description = article.find("p").text if article.find("p") else "No Description"
        data.append([title, description])

    df = pd.DataFrame(data, columns=["Title", "Description"])
    
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/fashion_trends.csv", index=False)
    print("Scraping Complete! Data saved in data/fashion_trends.csv")

scrape_fashion_trends()
