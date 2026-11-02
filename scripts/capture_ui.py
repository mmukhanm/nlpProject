"""Пять реальных browser-проверок и скриншотов запущенного Streamlit."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
REVIEWS = [
    'The sound quality is excellent and I love these headphones.',
    'The battery is terrible. This phone does not work.',
    'Great price and easy to use. Highly recommended.',
    'The charger broke after one day. Waste of money.',
    'The design is nice but the sound is poor and I cannot recommend it.',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8501')
    parser.add_argument('--browser', default='/usr/bin/chromium')
    args = parser.parse_args()
    folder = ROOT / 'reports/screenshots'; folder.mkdir(parents=True, exist_ok=True)
    rows = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=args.browser, headless=True, args=['--no-sandbox'])
        page = browser.new_page(viewport={'width': 1360, 'height': 1100}, device_scale_factor=1)
        for i, text in enumerate(REVIEWS, 1):
            page.goto(args.url)
            page.get_by_role('textbox', name='Отзыв покупателя').fill(text)
            page.get_by_role('button', name='Анализировать', exact=True).click()
            page.get_by_text('Вероятность выбранного класса', exact=True).wait_for(timeout=60000)
            # Wait for a completed result; locator asserts that a functional response exists.
            page.get_by_role('button', name='Скачать результат JSON').wait_for()
            label = page.locator('h3').first.inner_text()
            confidence = page.locator('[data-testid="stMetricValue"]').inner_text()
            page.screenshot(path=str(folder / f'control_{i:02d}.png'), full_page=True)
            rows.append({'text': text, 'sentiment': label, 'confidence': confidence, 'screenshot': f'control_{i:02d}.png'})
        browser.close()
    (ROOT / 'reports/ui_checks.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
