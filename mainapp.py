import streamlit as st
import pandas as pd
from scrapy.crawler import CrawlerRunner
from scrapy.utils.project import get_project_settings
from urllib.parse import urljoin
import scrapy
from twisted.internet import reactor
from twisted.internet.defer import inlineCallbacks
from twisted.internet.task import deferLater
from functools import partial
import json
from io import StringIO

# Store scraped data globally
scraped_data = []

class AmazonReviewsSpider(scrapy.Spider):
    name = "amazon_reviews"

    def __init__(self, asin=None, *args, **kwargs):
        super(AmazonReviewsSpider, self).__init__(*args, **kwargs)
        self.asin = asin

    def start_requests(self):
        amazon_reviews_url = f'https://www.amazon.com/product-reviews/{self.asin}/'
        yield scrapy.Request(url=amazon_reviews_url, 
                            callback=self.parse_reviews, 
                            meta={'asin': self.asin, 'retry_count': 0})

    def parse_reviews(self, response):
        asin = response.meta['asin']
        review_elements = response.css('div[data-hook="review"]')
        
        for review_element in review_elements:
            product_data = {
                "asin": asin,
                "text": "".join(review_element.css("span[data-hook=review-body] ::text").getall()).strip(),
                "title": review_element.css("*[data-hook=review-title]>span::text").get(),
                "location_and_date": review_element.css("span[data-hook=review-date] ::text").get(),
                "verified": bool(review_element.css("span[data-hook=avp-badge-linkless] ::text").get()),
                "rating": review_element.css("*[data-hook*=review-star-rating] ::text").re(r"(\d+\.*\d*) out")[0] if review_element.css("*[data-hook*=review-star-rating] ::text").re(r"(\d+\.*\d*) out") else None,
            }
            scraped_data.append(product_data)
            yield product_data

@inlineCallbacks
def run_spider(asin):
    global scraped_data
    scraped_data = []  # Clear previous data
    
    runner = CrawlerRunner(settings={
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'LOG_ENABLED': False,
        'ROBOTSTXT_OBEY': False,
        'DOWNLOAD_DELAY': 2,  # Add delay to avoid being blocked
    })
    
    yield runner.crawl(AmazonReviewsSpider, asin=asin)
    reactor.stop()

def sleep(_, *args, **kwargs):
    return deferLater(reactor, 3.5, lambda: None)

def scrape_complete(output, asin):
    st.session_state['scraping'] = False
    st.session_state['scraped_data'] = scraped_data
    st.experimental_rerun()

# Streamlit app
st.title("Amazon Reviews Scraper")

if 'scraping' not in st.session_state:
    st.session_state['scraping'] = False
if 'scraped_data' not in st.session_state:
    st.session_state['scraped_data'] = []

# Input ASIN
asin = st.text_input("Enter Amazon Product ASIN:", "B0D9HDH8ZV")

if st.button("Scrape Reviews") and not st.session_state['scraping']:
    st.session_state['scraping'] = True
    st.session_state['scraped_data'] = []
    
    # Run spider in a separate thread
    from threading import Thread
    thread = Thread(target=lambda: run_spider(asin).addCallback(scrape_complete, asin))
    thread.start()
    
    st.warning("Scraping in progress... Please wait.")

if st.session_state['scraping']:
    st.warning("Scraping in progress... Please wait.")

if st.session_state['scraped_data']:
    df = pd.DataFrame(st.session_state['scraped_data'])
    
    st.success(f"Successfully scraped {len(df)} reviews!")
    st.dataframe(df)
    
    # Download options
    csv = df.to_csv(index=False).encode('utf-8')
    json_data = StringIO()
    df.to_json(json_data, orient='records')
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name=f"amazon_reviews_{asin}.csv",
            mime="text/csv"
        )
    with col2:
        st.download_button(
            label="Download as JSON",
            data=json_data.getvalue(),
            file_name=f"amazon_reviews_{asin}.json",
            mime="application/json"
        )