from googlesearch import search
from bs4 import BeautifulSoup
import requests
import re
def fetch_top_search_results(query):
    """
    Perform a Google search and return structured information about the top search results.

    Args:
    - query (str): The search query.

    Returns:
    - results (list): A list of dictionaries containing titles, URLs, and snippets.
    """
    results = []
    try:
        search_results = search(query, num=3, stop=3, verify_ssl=False)
        for url in search_results:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.title.text if soup.title else "N/A"
            # Extract text from paragraphs and concatenate them
            paragraphs = soup.find_all('p')
            text = ' '.join([p.get_text() for p in paragraphs])
            # Remove special characters and extra whitespaces
            text = re.sub(r'\s+', ' ', text)  # Remove extra whitespaces
            text = re.sub(r'\[.*?\]', '', text)  # Remove text within square brackets
            text = re.sub(r'\([^()]*\)', '', text)  # Remove text within parentheses
            # Truncate text if it's too long
            text = text[:1500] + '...' if len(text) > 1500 else text
            result = f"Title: {title}\nURL: {url}\nText: {text}"
            results.append(result)
    except Exception as e:
        print("An error occurred:", e)
        
    print(results)
    
    return results

if __name__ == "__main__":
    while True:
        search_query = input("Enter your search query: ")
        
        if search_query.lower() == 'exit':
            break
        top_results = fetch_top_search_results(search_query)
        print("Top 5 search results for query:", search_query)
        results=[]
        for i, result in enumerate(top_results, start=1):
            result= f"{i}. {result}\n"
            results.append(result)
        print(results)
            
        
     