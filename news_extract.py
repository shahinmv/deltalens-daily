import requests
import sqlite3
from datetime import datetime, date
from tqdm import tqdm

# Cointelegraph API Configuration
API_URL = "https://conpletus.cointelegraph.com/v1/"
HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0"
}

QUERY = '''query TagPageQuery($short: String, $slug: String!, $order: String, $offset: Int!, $length: Int!) {
  locale(short: $short) {
    tag(slug: $slug) {
      posts(order: $order, offset: $offset, length: $length) {
        data {
          id
          slug
          postTranslate {
            title
            published
            leadText
          }
        }
        postsCount
      }
    }
  }
}'''

DB_PATH = '../db.sqlite3'
CUTOFF_DATE = datetime(2020, 5, 6).date()

class CointelegraphNewsScraper:
    def __init__(self):
        pass
    
    # === SHARED UTILITY FUNCTIONS ===
    def get_latest_news_date(self, db_path=DB_PATH):
        """Get the latest news date from the database"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(date) FROM news")
        result = cursor.fetchone()
        conn.close()
        if result and result[0]:
            return result[0]  # 'YYYY-MM-DD'
        return None
    
    def insert_news_to_db(self, news_items, db_path=DB_PATH, source="Cointelegraph"):
        """Insert news items into the SQLite database, handling duplicates properly"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        inserted = 0
        updated = 0
        
        for item in news_items:
            try:
                title = item['title']
                description = item.get('description', '')
                date_str = item.get('date', '')
                
                # Skip if essential data is missing
                if not title or not date_str:
                    continue
                    
                # Check if record already exists
                cursor.execute(
                    "SELECT COUNT(*) FROM news WHERE title = ? AND date = ?",
                    (title, date_str)
                )
                exists = cursor.fetchone()[0] > 0
                
                if exists:
                    # Update existing record
                    cursor.execute(
                        "UPDATE news SET description = ? WHERE title = ? AND date = ?",
                        (description, title, date_str)
                    )
                    if cursor.rowcount > 0:
                        updated += 1
                else:
                    # Insert new record
                    cursor.execute(
                        "INSERT INTO news (title, description, date) VALUES (?, ?, ?)",
                        (title, description, date_str)
                    )
                    if cursor.rowcount > 0:
                        inserted += 1
                        
            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed" in str(e):
                    # This is expected when trying to insert duplicates, skip silently
                    continue
                else:
                    print(f"Database integrity error processing {source} item: {item.get('title', 'Unknown')}\n  {e}")
            except Exception as e:
                print(f"Error processing {source} item: {item.get('title', 'Unknown')}\n  {e}")
        
        conn.commit()
        conn.close()
        print(f"{source}: Inserted {inserted} new items and updated {updated} existing items in the database.")
        return inserted, updated
    
    # === COINTELEGRAPH SCRAPER FUNCTIONS ===
    def fetch_cointelegraph_news(self, offset=0, length=15):
        """Fetch news from Cointelegraph API"""
        variables = {
            "short": "en",
            "slug": "bitcoin",
            "order": "postPublishedTime",
            "offset": offset,
            "length": length
        }
        payload = {
            "operationName": "TagPageQuery",
            "query": QUERY,
            "variables": variables
        }
        resp = requests.post(API_URL, json=payload, headers=HEADERS)
        resp.raise_for_status()
        return resp.json()
    
    def scrape_cointelegraph(self, latest_date=None):
        """Scrape Cointelegraph news starting from latest_date (inclusive)"""
        print(f"\n{'='*60}")
        print("STARTING COINTELEGRAPH SCRAPER")
        print(f"{'='*60}")
        print(f"Fetching Cointelegraph news from {latest_date or 'beginning'} onwards (inclusive)...")
        
        all_news = []
        offset = 0
        length = 15
        stop = False
        today = date.today()
        total_days = (today - CUTOFF_DATE).days
        earliest_date = today
        pbar = tqdm(total=total_days, desc="Collecting Cointelegraph news", unit="days")
        
        while not stop:
            data = self.fetch_cointelegraph_news(offset, length)
            posts = data['data']['locale']['tag']['posts']['data']
            if not posts:
                break
            for post in posts:
                title = post['postTranslate']['title']
                published = post['postTranslate']['published']
                description = post['postTranslate']['leadText']
                # Parse published date
                try:
                    pub_dt = datetime.fromisoformat(published.replace('Z', '+00:00')).date()
                    pub_date_str = pub_dt.strftime('%Y-%m-%d')
                except Exception:
                    pub_dt = None
                    pub_date_str = published[:10]
                
                # Stop when we encounter news OLDER than latest_date
                if latest_date and pub_date_str < latest_date:
                    print(f"Reached cutoff date. Stopping at {pub_date_str} (before {latest_date})")
                    stop = True
                    break
                
                # Include all news from latest_date onwards
                if not latest_date or pub_date_str >= latest_date:
                    all_news.append({
                        "title": title,
                        "time": published,
                        "description": description,
                        "date": pub_date_str
                    })
                
                # Update progress bar
                if pub_dt and pub_dt < earliest_date:
                    days_covered = (today - pub_dt).days
                    pbar.n = days_covered
                    pbar.refresh()
                    earliest_date = pub_dt
            offset += length
        pbar.n = total_days
        pbar.refresh()
        pbar.close()
        
        print(f"Cointelegraph: Collected {len(all_news)} news items")
        return all_news
    
    def run_scraper(self):
        """Run the Cointelegraph scraper"""
        print(f"\n{'='*80}")
        print("COINTELEGRAPH BITCOIN NEWS SCRAPER")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        
        # Get initial latest date
        initial_latest_date = self.get_latest_news_date()
        print(f"Latest news date in DB: {initial_latest_date}")
        
        total_inserted = 0
        total_updated = 0
        
        # === RUN COINTELEGRAPH SCRAPER ===
        try:
            cointelegraph_news = self.scrape_cointelegraph(latest_date=initial_latest_date)
            if cointelegraph_news:
                inserted, updated = self.insert_news_to_db(cointelegraph_news, source="Cointelegraph")
                total_inserted += inserted
                total_updated += updated
            else:
                print("Cointelegraph: No news items found")
        except Exception as e:
            print(f"Error in Cointelegraph scraper: {e}")
        
        # === FINAL SUMMARY ===
        final_latest_date = self.get_latest_news_date()
        print(f"\n{'='*80}")
        print("SCRAPING SUMMARY")
        print(f"{'='*80}")
        print(f"Initial latest date: {initial_latest_date}")
        print(f"Final latest date: {final_latest_date}")
        print(f"Total new items inserted: {total_inserted}")
        print(f"Total existing items updated: {total_updated}")
        print(f"Scraping completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")

def main():
    """Main function to run the Cointelegraph scraper"""
    scraper = CointelegraphNewsScraper()
    scraper.run_scraper()

if __name__ == "__main__":
    main()
    
    # === SHARED UTILITY FUNCTIONS ===
    def get_latest_news_date(self, db_path=DB_PATH):
        """Get the latest news date from the database"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(date) FROM news")
        result = cursor.fetchone()
        conn.close()
        if result and result[0]:
            return result[0]  # 'YYYY-MM-DD'
        return None
    
    def insert_news_to_db(self, news_items, db_path=DB_PATH, source="Unknown"):
        """Insert news items into the SQLite database, handling duplicates properly"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        inserted = 0
        updated = 0
        
        for item in news_items:
            try:
                # Handle different data structures
                if 'title' in item:
                    title = item['title']
                    description = item.get('description', '')
                    date_str = item.get('date', '')
                elif 'headline' in item:
                    title = item['headline']
                    description = item.get('description', '')
                    date_str = item.get('date', '')
                else:
                    continue
                
                # Skip if essential data is missing
                if not title or not date_str:
                    continue
                    
                # Check if record already exists
                cursor.execute(
                    "SELECT COUNT(*) FROM news WHERE title = ? AND date = ?",
                    (title, date_str)
                )
                exists = cursor.fetchone()[0] > 0
                
                if exists:
                    # Update existing record
                    cursor.execute(
                        "UPDATE news SET description = ? WHERE title = ? AND date = ?",
                        (description, title, date_str)
                    )
                    if cursor.rowcount > 0:
                        updated += 1
                else:
                    # Insert new record
                    cursor.execute(
                        "INSERT INTO news (title, description, date) VALUES (?, ?, ?)",
                        (title, description, date_str)
                    )
                    if cursor.rowcount > 0:
                        inserted += 1
                        
            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed" in str(e):
                    # This is expected when trying to insert duplicates, skip silently
                    continue
                else:
                    print(f"Database integrity error processing {source} item: {item.get('title', item.get('headline', 'Unknown'))}\n  {e}")
            except Exception as e:
                print(f"Error processing {source} item: {item.get('title', item.get('headline', 'Unknown'))}\n  {e}")
        
        conn.commit()
        conn.close()
        print(f"{source}: Inserted {inserted} new items and updated {updated} existing items in the database.")
        return inserted, updated
    
    # === COINTELEGRAPH SCRAPER FUNCTIONS ===
    def fetch_cointelegraph_news(self, offset=0, length=15):
        """Fetch news from Cointelegraph API"""
        variables = {
            "short": "en",
            "slug": "bitcoin",
            "order": "postPublishedTime",
            "offset": offset,
            "length": length
        }
        payload = {
            "operationName": "TagPageQuery",
            "query": QUERY,
            "variables": variables
        }
        resp = requests.post(API_URL, json=payload, headers=HEADERS)
        resp.raise_for_status()
        return resp.json()
    
    def scrape_cointelegraph(self, latest_date=None):
        """Scrape Cointelegraph news starting from latest_date (inclusive)"""
        print(f"\n{'='*60}")
        print("STARTING COINTELEGRAPH SCRAPER")
        print(f"{'='*60}")
        print(f"Fetching Cointelegraph news from {latest_date or 'beginning'} onwards (inclusive)...")
        
        all_news = []
        offset = 0
        length = 15
        stop = False
        today = date.today()
        total_days = (today - CUTOFF_DATE).days
        earliest_date = today
        pbar = tqdm(total=total_days, desc="Collecting Cointelegraph news", unit="days")
        
        while not stop:
            data = self.fetch_cointelegraph_news(offset, length)
            posts = data['data']['locale']['tag']['posts']['data']
            if not posts:
                break
            for post in posts:
                title = post['postTranslate']['title']
                published = post['postTranslate']['published']
                description = post['postTranslate']['leadText']
                # Parse published date
                try:
                    pub_dt = datetime.fromisoformat(published.replace('Z', '+00:00')).date()
                    pub_date_str = pub_dt.strftime('%Y-%m-%d')
                except Exception:
                    pub_dt = None
                    pub_date_str = published[:10]
                
                # Stop when we encounter news OLDER than latest_date
                if latest_date and pub_date_str < latest_date:
                    print(f"Reached cutoff date. Stopping at {pub_date_str} (before {latest_date})")
                    stop = True
                    break
                
                # Include all news from latest_date onwards
                if not latest_date or pub_date_str >= latest_date:
                    all_news.append({
                        "title": title,
                        "time": published,
                        "description": description,
                        "date": pub_date_str
                    })
                
                # Update progress bar
                if pub_dt and pub_dt < earliest_date:
                    days_covered = (today - pub_dt).days
                    pbar.n = days_covered
                    pbar.refresh()
                    earliest_date = pub_dt
            offset += length
        pbar.n = total_days
        pbar.refresh()
        pbar.close()
        
        print(f"Cointelegraph: Collected {len(all_news)} news items")
        return all_news