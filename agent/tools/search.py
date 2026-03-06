import requests
from langchain_core.tools import tool

@tool
def web_search(query: str) -> str:
    """
    Search the internet for current, real-time information.
    Use this when you need recent news, facts you are unsure about,
    current prices, weather, or anything that may have changed recently.
    Input should be a clear and specific search query.
    """
    try:
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no-html": "1",
            "skip_disambig": "1"
        }
        response = requests.get(url, params=params,timeout= 10)
        response.rais_for_status()
        data = response.json()

        results = []

        if data.get("AbstractText"):
            results.append(f"Summary: {data['AbstractText']}")
            if data.get("AbstractURL"):
                results.append(f"Summary: {data['AbstractURL']}")

        topics = data.get("RelatedTopics", [])[:3]
        for topic in topics:
            if isinstance(topic,dict) and topic.get("Text"):
               results.append(f"- {topic['Text']}")

        return "\n".join(results) if results else (
            f"No results found for '{query}'. Try a more specific query."
        )

    except requests.Timeout:
        return "Search timed out. Please try again."
    except Exception as e:
        return f"Search error: {str(e)}"
