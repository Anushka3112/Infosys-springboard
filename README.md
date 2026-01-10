```
# Deep Learning & Intelligent Web Scraping Projects

This repository contains multiple projects demonstrating **Deep Learning from scratch**, **MNIST prediction**, **CNN convolution demo**, **advanced web scraping**, and **semantic news-driven book pricing**.

---

## **MILESTONE 1 – Deep Neural Network (DNN) from Scratch on MNIST**

This project implements a **Deep Neural Network (DNN) from scratch** using **NumPy** to classify handwritten digits from the MNIST dataset.  
The goal is to study the impact of **learning rate**, **epochs**, and **model capacity** on training and validation performance.

### Project Overview

- **Dataset:** MNIST (Digits 0–9)  
- **Implementation:** Pure NumPy  
- **Architecture:** Fully Connected Neural Network  
- **Metrics:** Accuracy and Loss  
- **Visualization:** Training vs Validation Accuracy and Loss  

### Dataset Details

- **Training samples:** 60,000  
- **Validation samples:** 10,000  
- **Input size:** 784 (28×28 images flattened)  
- **Output classes:** 10  

#### Preprocessing

- Normalize pixels to `[0,1]`  
- Convert labels to **one-hot encoding**  

### Model Architecture

| Layer           | Neurons | Activation |
|----------------|---------|------------|
| Input Layer     | 784     | -          |
| Hidden Layer 1  | 128     | Sigmoid    |
| Hidden Layer 2  | 64      | Sigmoid    |
| Output Layer    | 10      | Softmax    |

### Implemented Features

- Forward propagation (Sigmoid & Softmax)  
- Backpropagation from scratch  
- Binary Cross-Entropy loss  
- Stochastic Gradient Descent (SGD)  
- Accuracy & loss tracking  
- Performance visualization using Matplotlib  

### Experiments & Analysis

#### Experiment 1

- **Architecture:** `[784,128,64,10]`  
- **Epochs:** 15  
- **Learning Rate:** 1.0  

**Observations:** Training accuracy ~100%, validation stagnates → overfitting.  

#### Experiment 2

- **Architecture:** `[784,128,64,10]`  
- **Epochs:** 50  
- **Learning Rate:** 0.5  

**Observations:** More stable learning, small gap between training & validation.  

#### Experiment 3 (Best Model)

- **Architecture:** `[784,128,64,10]`  
- **Epochs:** 200  
- **Learning Rate:** 0.01  

**Observations:** Smooth convergence, minimal gap, training >97%, validation >96%.  

### Visual Results

- **Training vs Validation Loss**  
- **Training vs Validation Accuracy**  

**Demonstrates:**  
- Overfitting at high learning rates  
- Stable convergence at low learning rates  
- Generalization behavior across epochs  

### How to Run

```bash
pip install numpy matplotlib scikit-learn tensorflow
python main.py

```bash
pip install numpy matplotlib scikit-learn tensorflow
python main.py


```bash
pip install numpy matplotlib scikit-learn tensorflow


```bash
pip install numpy matplotlib scikit-learn tensorflow
python main.py


# MNIST Digit Prediction Using Pre-Trained DNN

This project demonstrates how to **load a pre-trained Deep Neural Network (DNN)** trained on the **MNIST dataset** and use it to predict handwritten digits from custom input images.

The focus is on **MNIST-style preprocessing**, prediction confidence, and understanding **why a digit was predicted**.

---

## 📌 Project Overview

- Model: Pre-trained MNIST DNN (`mnist_dnn.keras`)
- Input: Handwritten digit image
- Output: Predicted digit (0–9) with confidence score
- Frameworks: Keras, NumPy, OpenCV
- Image size: 28×28 (flattened to 784 features)

---

## 🧠 Model Details

- The model was trained on **MNIST digits**
- Input shape: `(784,)`
- Output: Probability distribution over 10 digits
- Prediction is based entirely on **pixel similarity** to MNIST digits

⚠️ The model is **sensitive to input style and preprocessing**.

---

## 🖼️ Preprocessing Pipeline (MNIST-Style)

The input image is transformed to closely match MNIST training data:

1. **Grayscale conversion**
2. **Binary thresholding & inversion**
   ```python
   _, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

Digit → white

Background → black

Removes unnecessary background noise

Digit extraction

Finds bounding box using contours

Resize while preserving aspect ratio

Digit scaled to fit within 20×20

Centering

Digit placed in the center of a 28×28 black canvas

Normalization

Pixel values scaled to [0, 1]

Flattening

Reshaped to (1, 784) for DNN input

Why This Digit Was Predicted

The input digit most closely resembles the digit 8 in the model’s learned feature space

Flattened DNNs rely heavily on pixel-level patterns

Even small changes in:

centering

thickness

inversion

rotation
can influence predictions

⚠️ Limitations of MNIST-Trained DNNs

Trained only on clean MNIST digits

Sensitive to:

handwriting style

stroke thickness

misalignment

preprocessing mismatch

Real-world handwritten digits may differ at pixel level

📌 The model predicts the closest-looking digit, not necessarily the true digit.


# CNN Convolution & ReLU Demonstration Using Sobel Filters

This project demonstrates **how a Convolutional Neural Network (CNN) convolution layer works internally**, using **Sobel edge-detection filters** and **ReLU activation**, implemented **from scratch using NumPy**.

The goal is to visualize:
- How convolution extracts features
- How ReLU affects feature maps
- How edges are detected in images

---

## 📌 Project Overview

- Implementation: Pure NumPy (no deep learning frameworks)
- Image processing: OpenCV
- Visualization: Matplotlib
- Filters used: Sobel Gx & Gy
- Activation: ReLU (Rectified Linear Unit)

---

## 🧠 Concept Explained

### 1️⃣ Convolution in CNNs

A CNN convolution layer:
- Slides a small kernel (filter) over the image
- Performs element-wise multiplication
- Sums the result to produce a feature map

This project uses **3×3 Sobel filters** to detect edges.

---

### 2️⃣ Sobel Filters Used

#### Vertical Edge Detection (Gx)
[-1 0 1
-2 0 2
-1 0 1]


#### Horizontal Edge Detection (Gy)
[-1 -2 -1
0 0 0
1 2 1]


## ⚙️ CNN Convolution Layer (From Scratch)

### Key Steps:
1. Zero padding to preserve spatial size
2. Sliding 3×3 window across the image
3. Dot product between patch and filter
4. Apply **ReLU activation**

```python
output[i, j, f] = max(0, raw)

Why ReLU Is Important

Introduces non-linearity

Removes negative activations

Helps CNNs learn meaningful features

Improves training stability

This demo shows how ReLU suppresses weak or irrelevant edges.

Output Feature Maps

The script visualizes:

Original grayscale image

Raw Gx output (vertical edges)

Raw Gy output (horizontal edges)

Final edge magnitude

magnitude = sqrt(Gx² + Gy²)


This mimics early CNN layers used in real vision models.


** Milestone 2: Advanced Web Scraping & Data Processing Pipelines **

This milestone demonstrates **multiple real-world web scraping approaches** using modern tools and APIs.  
It focuses on **dynamic scraping, structured API extraction, and intelligent data enrichment**.

---

## 📌 Milestone Objectives

- Scrape **dynamic websites** using Playwright
- Extract **structured e-commerce data** using ScraperAPI
- Scrape **static websites** using Requests + BeautifulSoup
- Apply **business logic (pricing & sentiment analysis)**
- Store data in **CSV, JSON, and Excel formats**

---

## 🧩 Tools & Technologies Used

| Tool / Library | Purpose |
|----------------|--------|
| Playwright | Dynamic website scraping |
| Chromium | Headless browser |
| nest_asyncio | Async execution in notebooks |
| Requests | HTTP requests |
| BeautifulSoup | HTML parsing |
| Pandas | Data processing & export |
| OpenCV / NumPy | (Not used here) |
| ScraperAPI | Amazon structured data |
| CSV / JSON | Data storage |

---

## 🚀 PART 1: Dynamic Website Scraping (Playwright)

### Target Website
**WebScraper Test E-Commerce Site**

### What This Code Does
- Automatically detects **total pages**
- Scrapes **all product cards**
- Extracts:
  - Title
  - Price
  - Rating (stars)
  - Image URL
  - Page number
- Saves results in **CSV and JSON**

### Key Features
- ❌ No timeouts
- ❌ No retries
- ✅ Fully async
- ✅ Pagination handled dynamically

### Core Logic
1. Launch headless Chromium
2. Detect total pages via pagination
3. Loop through all pages
4. Extract product data from `.thumbnail`
5. Save output to `output/`

### Output Files
output/all_products.csv
output/all_products.json


---

## 📦 PART 2: Amazon Product Scraping (ScraperAPI)

### Why ScraperAPI?
Amazon blocks direct scraping.  
ScraperAPI provides **structured Amazon data legally and safely**.

### What This Code Does
- Uses ScraperAPI structured endpoint
- Accepts **keyword input**
- Automatically paginates
- Extracts:
  - Title
  - Price
  - ASIN
  - Rating
  - Reviews
  - Keyword used

### Key Logic
1. Call ScraperAPI Amazon Search endpoint
2. Fetch page-by-page until results stop
3. Append keyword metadata
4. Save data in **JSON, CSV, Excel**

### Output Files
output/amazon_products.json
output/amazon_products.csv
output/amazon_products.xlsx


### Example Summary
Keyword: laptops
Total Products: 120
Total Pages: 6

yaml
Copy code

---

## 📚 PART 3: Books Data Scraping + Smart Pricing

### Target Website
https://books.toscrape.com


### What This Code Does
- Scrapes **all book listings**
- Visits **each detail page**
- Extracts:
  - Title
  - Original price
  - Rating
  - Stock availability
  - Description
- Applies **pricing strategy**
- Adds business flags

---

## 🧠 Smart Logic Implemented

### ⭐ Rating Conversion
Text → Numeric


One → 1
Two → 2
...
Five → 5


### 💬 Sentiment Analysis (Rule-Based)
- Positive words → price boost
- Negative words → no boost
- Score range: `-1 to +1`

### 💰 Pricing Strategy
| Condition | Price Change |
|---------|-------------|
| 5-Star Book | +7% |
| Low Stock (≤5) | +5% |
| Positive Sentiment | Small boost |

### 🔥 Business Flags
- `hot_selling` → High rating + low stock
- `last_copy` → Only 1 book left

---

## 📊 Final Output Files



output/books_prices.csv
output/books_prices.xlsx
output/books_prices.json


---


🧠 Semantic News-Driven Book Pricing System
**Milestone 3 – NLP, Semantic Similarity & Dynamic Pricing**
📌 Project Overview

This milestone implements an intelligent book pricing system that dynamically adjusts book prices using:

📰 Live news sentiment

🧠 Semantic similarity using NLP embeddings

📅 Seasonal pricing strategies

⭐ Book ratings

The system scrapes 1,000 books from Books to Scrape, matches them with current news headlines, analyzes sentiment, and adjusts prices based on real-world context.

🎯 Key Features

Semantic matching between book descriptions and news headlines

Sentiment-driven price adjustment

Seasonal demand rules

Rating-based price boost

Fully automated pipeline

CSV output for further analysis

🛠️ Installation & Setup
Libraries Used

sentence-transformers → Semantic embeddings

nltk → Text preprocessing & sentiment analysis (VADER)

requests → Web requests

beautifulsoup4 → Web scraping

pandas → Data processing

torch → Tensor similarity computation

Installation
pip install sentence-transformers nltk requests beautifulsoup4

🧩 NLTK Setup (Text & Sentiment Processing)
Resources Downloaded

WordNet → Lemmatization

Stopwords → Noise removal

VADER → Sentiment analysis

Core Functions
clean_text(text)

Removes stopwords

Keeps meaningful tokens

Lemmatizes words

sentiment_score(text)

Uses VADER compound score

Output:

1 → Positive

0 → Neutral

-1 → Negative

📰 Fetching Live News (Bing RSS)
News Source
https://www.bing.com/news/search?q=latest+news&format=rss

Process

Fetches latest news headlines

Extracts:

Title

Link

Publish date

Sentiment score

📦 Stored in a Pandas DataFrame: news_df

🧠 Semantic Embedding Model
Model Used
all-MiniLM-L6-v2

Purpose

Converts news headlines into vector embeddings

Enables semantic similarity matching with book descriptions

Captures meaning beyond keyword matching

📚 Book Scraping (1,000 Books)
Source Website
https://books.toscrape.com

Data Collected per Book
Title
Price
Rating (1–5)
Stock availability
Category
Description
Scraping Strategy
Iterates through all pages
Visits individual book pages
Extracts detailed metadata

🔗 Semantic Matching with News
Matching Logic

Clean book description

Generate embedding

Compare with all news embeddings

Select most semantically similar news headline

New Columns Added

matched_news

news_sentiment

📌 This connects real-world events to book themes.

📅 Seasonal & Strategic Pricing Rules
Season / Theme	Keywords	Multiplier
Holiday	gift, christmas	1.20
Vacation	travel, summer	1.15
Education	school, study	1.18
Politics	war, government	1.12
Rule Application

Keywords are matched against:

Book category

Book description

Related news headline

💰 Dynamic Price Adjustment Logic
Pricing Factors
1️⃣ News Sentiment Impact

Positive → +10%

Negative → −5%

2️⃣ Rating Bonus

Rating ≥ 4 → +15%

3️⃣ Seasonal Multiplier

Based on detected season/theme

Final Pricing Formula
Adjusted Price = Base Price × Sentiment Factor × Rating Factor × Seasonal Multiplier

📤 Output
Generated File
books_dynamic_pricing_news.csv

CSV Contains

Book title

Original price

Adjusted price

Rating

Category

Matched news headline

News sentiment

Seasonal reason

🧠 Why This Milestone Is Important

Demonstrates real-world NLP application

Combines scraping, embeddings, sentiment & business logic

Shows how AI can drive dynamic pricing strategies

Bridges data science + decision systems

🚀 Future Enhancements

Real-time news API integration

Reinforcement learning for price optimization

part 2 - author popularity , reviews 

##  Features

- Scrapes **1000 books** from [BooksToScrape](https://books.toscrape.com) including title, price, rating, and description.
- Fetches **author data** from Open Library API (author name and edition count).
- Performs **sentiment analysis** on book descriptions.
- Computes **cosine similarity** between book descriptions and **latest news headlines**.
- Applies **seasonal strategies** to increase pricing during relevant seasons (e.g., Christmas gifts, educational books, political topics).
- Combines multiple factors to compute a **final demand score** for each book.
- Calculates **adjusted prices** dynamically based on demand, sentiment, and seasonal factors.
- Outputs a CSV file with **all enriched book data** and price recommendations.

---

## 🛠️ Libraries & Tools Used

| Library | Purpose |
|---------|---------|
| `requests` | HTTP requests for scraping and API calls |
| `pandas` | Data storage, transformation, and output |
| `BeautifulSoup` | HTML parsing for web scraping |
| `nltk` | Text preprocessing (stopwords, lemmatization) |
| `vaderSentiment` | Sentiment scoring of book descriptions |
| `sentence_transformers` | Semantic embeddings for text similarity |
| `torch` | Tensor computations and cosine similarity |
| `tqdm` | Progress bars for loops |

---

##  Pipeline Steps

### 1️⃣ NLTK Setup

- Download stopwords and WordNet lemmatizer.
- Initialize:
  - `stop_words` → words to ignore for text processing.
  - `lemmatizer` → reduces words to base forms.
  - `sentiment_analyzer` → scores text positivity/negativity.

---

### 2️⃣ Text Cleaning & Sentiment

- **clean_text**: removes non-alphabetic characters, words <4 letters, stopwords, and lemmatizes text.
- **get_sentiment**: returns a compound sentiment score (-1 to 1).
- **sentiment_label**: converts numeric score into "Positive", "Neutral", or "Negative".

---

### 3️⃣ Fetch Latest News

- Scrapes **top 10 news titles** from Bing News RSS feed.
- News embeddings will be used to determine **book relevance to current events**.

---

### 4️⃣ Load Embedding Model

- Uses `SentenceTransformer("all-MiniLM-L6-v2")` to encode both:
  - **News headlines**
  - **Book descriptions**
- Enables **semantic similarity matching** between books and news.

---

### 5️⃣ Scrape Books Data

- Scrapes **BooksToScrape** (1000 books across multiple pages):
  - `title`, `price`, `rating`, `description`
- Converts **ratings from text to numeric**:
  - `"One" → 1, "Two" → 2, ..., "Five" → 5`

---

### 6️⃣ Fetch Author Data

- Uses **Open Library API** to fetch author name and edition count for each book title.
- Computes **author popularity index** based on:
  1. Number of books by author
  2. Average rating of their books
  3. Average edition count
- Formula for `author_popularity_index`:
author_popularity_index = 0.4 * normalized(book count) +
0.4 * normalized(avg rating) +
0.2 * normalized(avg editions)



---

### 7️⃣ Review & Sentiment Scoring

- Uses **book description as proxy for reviews**.
- Computes:
  - `review_length` → length of description
  - `sentiment_score` → compound sentiment
  - `sentiment_label` → Positive/Neutral/Negative

---

### 8️⃣ Cosine Similarity with News

- Computes similarity between **cleaned book description** and **top news embeddings**.
- Adds:
  - `matched_news` → most similar news headline
  - `news_similarity_score` → cosine similarity score

---

### 9️⃣ Seasonal Strategy

- Applies **seasonal multipliers** for price adjustment:
  - `"Holiday"` → 1.2 (keywords: christmas, gift)
  - `"Education"` → 1.15 (keywords: school, study)
  - `"Politics"` → 1.1 (keywords: war, government)
- Adds columns:
  - `seasonal_reason` → detected season
  - `seasonal_multiplier` → multiplier value

---

### 🔟 Compute Final Demand Score

- Combines multiple factors for **overall book demand**:

final_demand_score = 0.25 * sentiment_score +
0.20 * (review_length / max(review_length)) +
0.20 * news_similarity_score +
0.20 * author_popularity_index +
0.15 * (rating / 5)


- Flag books in **top 10%** demand:
df["high_demand"] = df["final_demand_score"] >= df["final_demand_score"].quantile(0.9)


---

### 1️⃣1️⃣ Calculate Adjusted Price

- Adjusted price formula:




- Multiplies original price by **demand factor** and **seasonal multiplier**.

---

### 1️⃣2️⃣ Save Output

- Saves **enriched dataset** to CSV:


**Milestone 4 - cross platform Integration and Notification Deployment system **

📌 Introduction

In competitive online book marketplaces, manual price tracking is inefficient and error-prone. Prices change frequently, and delayed decisions can lead to revenue loss.

This project provides a fully automated system that:

Collects book pricing data from multiple sources

Matches books accurately using ISBN-13

Analyzes competitor prices

Suggests an optimized selling price using a defined pricing strategy

The system outputs structured data in CSV format for further business analysis.

🎯 Project Objectives

Automate book price collection from multiple platforms

Ensure accurate book matching using ISBN-13 and title similarity

Analyze competitor pricing in real time

Recommend competitive, profit-oriented pricing

Reduce manual effort and pricing decision lag

🔍 Data Sources Used
1️⃣ BooksToScrape

Source of our store’s book prices

Used for scraping:

Book title

Retail price

2️⃣ Google Books API

Used to:

Validate book identity

Fetch ISBN-13 numbers

Ensures accurate competitor matching

3️⃣ BooksRun API

Used to retrieve competitor prices:

New books

Used books

Rental prices

Marketplace used prices

🧠 System Architecture
BooksToScrape
     ↓
(Book Title + Our Price)
     ↓
Google Books API
     ↓
(ISBN-13 Matching)
     ↓
BooksRun API
     ↓
(Competitor Prices)
     ↓
Pricing Strategy Engine
     ↓
CSV Output

🛠️ Technology Stack

Python 3

Requests – API calls

BeautifulSoup – Web scraping

Pandas – Data processing & CSV export

Difflib (SequenceMatcher) – Title similarity matching

REST APIs – Google Books & BooksRun

⚙️ Configuration Parameters
Parameter	Description
TOTAL_PAGES	Number of pages scraped from BooksToScrape
SIMILARITY_THRESHOLD	Title matching accuracy (default: 0.7)
SLEEP_TIME	Delay between requests (API safety)
BOOKSRUN_API_KEY	BooksRun API authentication key
OUTPUT_FILE	Name of generated CSV file
🧩 Core Logic Explanation
1️⃣ Scraping Our Prices

Scrapes multiple pages from BooksToScrape

Extracts:

Book title

Price (cleaned to float format)

our_price = clean_price(price_text)

2️⃣ Title Cleaning & Matching

To avoid incorrect matches:

Titles are cleaned (lowercase, symbols removed)

Compared using SequenceMatcher

Only titles with similarity ≥ 0.7 are accepted

similarity(a, b) >= SIMILARITY_THRESHOLD

3️⃣ ISBN-13 Extraction

Google Books API returns multiple identifiers

Only ISBN-13 is used for competitor lookup

Ensures consistent and accurate pricing data

4️⃣ Competitor Price Selection

BooksRun API may return multiple prices:

New

Used

Rental

Marketplace used

The lowest valid competitor price is selected:

competitor_price = min(valid_prices)

📊 Market-Adjusted Pricing Strategy
Strategy Rules

If competitor price > our price
→ Increase price, but still undercut competitor by 5%

If competitor price ≤ our price
→ Reduce price by 5% to stay competitive

Formula
adjusted_price = competitor_price × (1 − 0.05)
profit = adjusted_price − our_price


This ensures:

Competitive pricing

Higher chance of sales

Predictable margin estimation

📁 Output Structure

The script generates a CSV file:

booksrun_price_comparison.csv which has upladed in output fold

##  Sample Columns
Column             	Description
title             	Book title
isbn13	            ISBN-13 identifier
our_price	         Original store price
competitor_price  	Lowest competitor price
used_price	         Used book price
new_price	         New book price
rental_price       	Rental price
marketplace_price 	Marketplace used price
pricing_action	      Increase / Reduce
adjusted_price	      Recommended price
profit            	Expected profit

▶️ How to Run
1️⃣ Install Dependencies
pip install requests pandas beautifulsoup4

2️⃣ Execute Script
python main.py

3️⃣ View Results

Open:

booksrun_price_comparison.csv

📈 Business Use Cases

E-commerce price optimization

Automated competitor analysis

Retail pricing intelligence

Academic / final-year project

Resume-ready data engineering project

🚀 Future Enhancements

Dynamic pricing based on demand & sales history

Multiple competitor integration

Machine learning price prediction

Web dashboard visualization

Real-time price alerts

⚠️ Notes

Uses rate limiting to avoid API blocking

Prevents duplicate ISBN processing

Designed for scalability and modular expansion











