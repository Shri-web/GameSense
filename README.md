

# GameSense: An E-Commerce Recommendation System

**GameSense** is an advanced e-commerce recommendation system that leverages both tabular data and semantic search capabilities to provide users with highly relevant product recommendations. By integrating the power of PostgreSQL for robust data storage and management, Pinecone for vector-based similarity searches, and Cohere for sophisticated re-ranking, GameSense aims to deliver an enhanced user experience for finding and exploring products, particularly focusing on gaming-related items.

## Features

### Hybrid Search Capabilities
- **Tabular Search**: Allows users to filter products based on categories, brands, operating systems, price range, and ratings.
- **Semantic Search**: Enables users to enter natural language queries to find products that match their search intent through semantic analysis of product reviews.

### Intelligent Recommendations
- **Combined Re-Ranking**: Utilizes Cohere's re-ranking models to sort search results by relevance, ensuring the most relevant products are displayed first.
- **Top Review Highlights**: Highlights the most relevant review for each product based on the user's search query, enhancing decision-making.

### User-Friendly Interface

- **Detailed Product View**: Allows users to view detailed information about each product, including specifications, price, and top reviews.


## Technologies Used

- **Python**: Core programming language for backend and frontend development.
- **Streamlit**: Framework for building the interactive web interface.
- **PostgreSQL**: Database system for storing product data.
- **Pinecone**: Vector database for semantic search capabilities.
- **Cohere**: API for semantic re-ranking of search results.
- **OpenAI**: For generating embeddings from user queries.
- **tiktoken**: Tokenizer for handling input text.

## Getting Started

### Prerequisites

Ensure you have the following software installed on your machine:
- Python 3.7+
- Streamlit

### Installation

1. **Clone the Repository**:
    ```bash
    https://github.com/Shri-web/GameSense.git
    cd GameSense
    ```

2. **Install Required Packages**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Configure Environment Variables**:
    - Set up environment variables for API keys (Cohere, Pinecone, OpenAI) and database credentials.

5. **Run the Application**:
    ```bash
    streamlit run app.py
    ```

## Usage

- **Home Page**: Browse the most popular products.
- **Search**: Enter a query or use the sidebar filters to find specific products.
- **View Product**: Click on a product to view detailed information and top reviews.
## Images



## Acknowledgements

- [PostgreSQL](https://www.postgresql.org/)
- [Streamlit](https://streamlit.io/)
- [Pinecone](https://www.pinecone.io/)
- [Cohere](https://cohere.ai/)
- [OpenAI](https://www.openai.com/)

---

Feel free to customize this description to better fit your project's specific details and any additional features you plan to implement.
