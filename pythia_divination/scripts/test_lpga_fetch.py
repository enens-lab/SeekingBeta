import requests
from bs4 import BeautifulSoup

def test_lpga_stats():
    url = "https://www.lpga.com/statistics"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        print(f"Status Code: {response.status_code}")
        print(f"Content Length: {len(response.text)}")
        if len(response.text) > 0:
            soup = BeautifulSoup(response.text, "html.parser")
            table = soup.find("table")
            if table:
                print("Table found!")
                print(str(table)[:500])
            else:
                print("No table found.")
                # Check for script tags
                scripts = soup.find_all("script")
                print(f"Number of scripts: {len(scripts)}")
                for s in scripts:
                    if s.string and "window.__INITIAL_STATE__" in s.string:
                        print("Found __INITIAL_STATE__!")
                        break
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_lpga_stats()
