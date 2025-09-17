from playwright.sync_api import sync_playwright
import pandas as pd

def scrape_latest_draw():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("https://example-keno-site.com/results")  # Replace with actual URL

        # Adjust selector to match actual site structure
        draw_selector = ".draw-result .number"
        numbers = page.locator(draw_selector).all_text_contents()
        numbers = [int(n.strip()) for n in numbers if n.strip().isdigit()]

        if len(numbers) == 20:
            df = pd.DataFrame([numbers])
            df.to_csv("data/draws.csv", mode='a', header=False, index=False)
            print("✅ New draw appended:", numbers)
        else:
            print("⚠️ Draw incomplete or invalid:", numbers)

        browser.close()
