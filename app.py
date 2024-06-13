import streamlit as st
import pandas as pd
import numpy as np
from pinecone import Index
import cohere
from sentence_transformers import SentenceTransformer # Add the missing import statement for load_dotenv
import time
import random
from langchain.memory import ConversationSummaryMemory,ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.prompts import HumanMessagePromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY_1 = os.getenv("PINECONE_API_KEY_1")
PINECONE_API_KEY_2 = os.getenv("PINECONE_API_KEY_2")
PINECONE_INDEX_NAME_1 = os.getenv("PINECONE_INDEX_NAME_1")
PINECONE_INDEX_NAME_2 = os.getenv("PINECONE_INDEX_NAME_2")
PINECONE_HOST_1 = os.getenv("PINECONE_HOST_1")
PINECONE_HOST_2 = os.getenv("PINECONE_HOST_2")
COHERE_API_KEY = os.getenv("COHERE_KEY")
PATH_DV = os.getenv("PATH_DV")


@st.cache_resource
def load_groq():
    groq_api_key = GROQ_API_KEY
    return groq_api_key

@st.cache_resource
def load_pinecone_index():
    pinecone_api_key = PINECONE_API_KEY_1
    pinecone_index_name = PINECONE_INDEX_NAME_1
    return Index(index_name=pinecone_index_name, api_key=pinecone_api_key, host=PINECONE_HOST_1)
@st.cache_resource
def load_pincone_index2():
    pinecone_api_key = PINECONE_API_KEY_2
    pinecone_index_name = PINECONE_INDEX_NAME_2
    return Index(index_name=pinecone_index_name, api_key=pinecone_api_key, host=PINECONE_HOST_2)
@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer('all-mpnet-base-v2')

@st.cache_resource
def load_cohere_client():
    return cohere.Client(COHERE_API_KEY)

@st.cache_resource
def load_data():
    df = pd.read_csv(PATH_DV)
    return df

def get_pinecone_results_reviews(query, model, index,asin):
    query_embedding = model.encode(query).tolist()
    pinecone_results = index.query(vector=query_embedding, top_k=8, filter = { "asin": {"$eq": asin}},include_metadata=True)
    return pinecone_results['matches']

def get_pinecone_results(query, model, index):
    query_embedding = model.encode(query).tolist()
    pinecone_results = index.query(vector=query_embedding, top_k=15, include_metadata=True)
    return pinecone_results['matches']

def rerank_with_cohere(query, descriptions, co_client):
    rerank_response = co_client.rerank(
        query=query,
        documents=[{"text": doc} for doc in descriptions],
        top_n=5,
        model="rerank-multilingual-v2.0"
    )
    return rerank_response.results

def set_css():
    """ Set custom CSS for the app. """
    st.markdown("""
    <style>
    .image-container {
        height: 200px;
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 5px;
    }
    .image-container img {
        max-height: 100%;
        max-width: 100%;
    }
    .info {
        text-align: center;
        font-size: 16px;
        font-weight: bold;
        color: #4f8bf9;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

def chat_response(qa,asin):
            data = load_data()
            search_query = qa  # Strip whitespace from the search query
            model = load_sentence_transformer()
            pinecone_index_2 = load_pincone_index2()
            co_client = load_cohere_client()
            pinecone_results = get_pinecone_results_reviews(search_query, model, pinecone_index_2, asin)
            review = [res['metadata']['reviewTxt'] for res in pinecone_results]
            product_description = data[data['asin']==asin]['description_str']
            title = data[data['asin']==asin]['title']
            console = data[data['asin']==asin]['Category']
            brand = data[data['asin']==asin]['brand']
            context = f"**USER QUESTION:**\n{qa}\n\n"
            if review == []:
                
                context += f"**Title:**\n{title}\n\n**Console**\n{console}\n\n**Brand**\n{brand}\n\n**PRODUCT DESCRIPTION:**\n{product_description}\n\n"
            elif search_query == '' or search_query.isspace():
                
                 context += f"**Title:**\n{title}\n\n**Console**\n{console}\n\n**Brand**\n{brand}\n\n**PRODUCT DESCRIPTION:**\n{product_description}\n\n**REVIEW:**\n{review}\n)\n\n"
            else:
                cohere_results = rerank_with_cohere(search_query, review, co_client)
                top_results = [review[res.index] for res in cohere_results]
                context += f"**Title:**\n{title}\n\n**Console**\n{console}\n\n**Brand**\n{brand}\n\n**PRODUCT DESCRIPTION:**\n{product_description}\n\n**REVIEW:**\n{top_results}\n\n"

            context = context[:7700]  # Limit the context to 7700 characters

            context += "\n**RESPONSE TO DISPLAY TO THE USER:**"
            prompt = f"""
            You are a helpful assistant that reads Amazon reviews of a videogame product and answers a question for an Amazon user. 
If the question is about video games generally, ignore the reviews and provide a general response by looking into the description.
If the question is product-specific, perform the following steps:  
    
1) Read the following context then in a markdown bulleted list, provide 0-5 direct quotes from the reviews that are relevant to the user's question .
2) In a short paragraph, respond to the user's question based on the reviews. You may refer to the product description as well if it contains information that answers the user's question, but focus primarily on conveying relevant information from the reviews whenever possible. 
3) If there is no review look into the product description and provide a response based on that.And mention that there is no review.
4) If there is no review or product description, respond with the exact phrase "Sorry, I don't know".
    
Try to generalize the information in the reviews - do not rehash the provided review snippets verbatim (as the user will have read them); just use them to provide a concise, measured answer to the user's question. 
Strive to provide a response, but do not make up information. If the question is unrelated to computers or the product, respond with the exact phrase "Sorry, I don't know".
*CRITICALLY, the response you provide is being displayed directly to the user who asked the question -- RESPOND DIRECTLY TO THE USER.*

            
            
            {context}"""

            groq_api_key = load_groq()
            groq_chat = ChatGroq(
                groq_api_key=groq_api_key,
                model_name='llama3-70b-8192'
            )
            memory=ConversationBufferWindowMemory(k=5)
            
            chat_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=(
                        prompt
                    )
                ),
                HumanMessagePromptTemplate.from_template("{text}"),
             ])
            
            messages = chat_template.format_messages(text=qa)

            llm =  ChatGroq(
            groq_api_key=groq_api_key, 
            model_name='mixtral-8x7b-32768')

            conversation = ConversationChain(
                llm=groq_chat,
                memory=memory,
            )
            response = conversation(messages)

            st.markdown(f"**Chatbot Response:**\n{response.get('response')}", unsafe_allow_html=True)


def view_product_details(product_id, data):
    if 'product' in st.session_state and st.session_state.product != '' and \
        'page' in st.session_state and st.session_state.page == 'view':

        if st.button('Back to Search', key='back_tab1'):
            del st.session_state['product']
            st.session_state.popped=True
            st.session_state.page = 'search'
            st.rerun()

        tab1, tab2 ,tab3= st.tabs(['Product','Top Reviews' ,'Chat with Reviews'])
        with tab1:
            product = data[data['asin'] == product_id].iloc[0]
            st.title(product['title'])
            st.write(f"Brand: {product['brand']}")
            st.markdown(product['description_str'], unsafe_allow_html=True)
            st.write(f"Category: {product['Category']}")
            print(product_id)
            # Parse and display in-game pictures
            in_game_pictures = eval(product['in_game_pictures'])
            if in_game_pictures:
                img_urls = in_game_pictures[0]  # Take only the first set of images
                if img_urls:
                    st.subheader("In-game Pictures:")
                    for img_url in img_urls:
                        st.image(img_url)
                else:
                    st.write("No images available.")
        with tab2:
            search_query = st.session_state.query_for_sorting.strip()  # Strip whitespace from the search query
            model = load_sentence_transformer()
            pinecone_index_2 = load_pincone_index2()
            co_client = load_cohere_client()
            pinecone_results = get_pinecone_results_reviews(search_query, model, pinecone_index_2, product_id)
            review = [res['metadata']['reviewTxt'] for res in pinecone_results]
            if review == []:
                st.write('No reviews found')
            elif search_query == '' or search_query.isspace():
                for i,j in enumerate(review):
                    st.markdown(f"**Review {i+1}:** {j}", unsafe_allow_html=True)
            else:
                cohere_results = rerank_with_cohere(search_query, review, co_client)
                top_results = [review[res.index] for res in cohere_results]
                for i,j in enumerate(top_results):
                    st.markdown(f"**Review {i+1}:** {j}", unsafe_allow_html=True)
        with tab3:
            st.title("Reviews Chat")
            # Create a form
            with st.form(key='chat_form'):
                qa = st.text_input("Ask a question:")
                submit_button = st.form_submit_button("Submit")

            # Process the form when the submit button is clicked
            if submit_button:
                if qa:
                    
                    chat_response(qa, product_id)




def set_viewed_product(product_id):
    st.session_state.product = product_id
    st.session_state.page = 'view'
    st.session_state.from_reload = True



def header(header_text):
    st.markdown(f'<p style="font-size:42px;font-weight:bold;font-family:sans-serif;color:#ffb86c;">{header_text}</p>', unsafe_allow_html=True)
def subheader(header_text):
    st.markdown(f'<p style="font-size:18px;font-family:sans-serif;color:#50fa7bff;">{header_text}</p>', unsafe_allow_html=True)
# endregion CSS

def view_products(data, products_per_row=4):
    if 'product' not in st.session_state and st.session_state.page == 'search':
        if (st.session_state.from_reload) or ('popped' not in st.session_state or st.session_state.popped == False):
            header('GameSpace')
            subheader('Search for your favorite video games')

            num_rows = int(np.ceil(len(data) / products_per_row))
            for i in range(num_rows):
                start = i * products_per_row
                end = start + products_per_row
                products = data.iloc[start:end]

                columns = st.columns(products_per_row)
                for column_index, product in enumerate(products.iterrows()):
                    asin = product[1]['asin']
                    button_key = f"view_{asin}"  # Unique button key
                    container = columns[column_index].container()
                    container.image(product[1]['first_url'], use_column_width='always', caption=product[1]['title'])
                    container.markdown(f"**Title** : {product[1]['title']}")
                    container.write(f"**Brand** : {product[1]['brand']}")
                    container.write(f"**Category** : {product[1]['Category']}")
                    if container.button('View Details', key=button_key):
                        set_viewed_product(product_id=asin)

            st.session_state.popped = True


def recommend_games():
    data = load_data()
    search_query = st.session_state.query_for_sorting.strip()  # Strip whitespace from the search query
    model = load_sentence_transformer()
    pinecone_index = load_pinecone_index()
    co_client = load_cohere_client()

    if search_query == '' or search_query.isspace():  # Check if the search query is empty or only whitespace
        st.session_state.filtered_products_df = data.sample(10, random_state=5)  # Sample random 10 products
        view_products(st.session_state.filtered_products_df)
        st.session_state.popped = True  # Set popped state to True after displaying products
        st.session_state.from_reload = False
        return  # Exit the function if no search query is provided

    if (st.session_state.get('FormSubmitter:filter_form-Search', False)) or (st.session_state.get('filtered_products_df', None) is None):
        with st.spinner('Searching...'):
            time.sleep(1)

        if ('FormSubmitter:filter_form-Search' in st.session_state and st.session_state['FormSubmitter:filter_form-Search']) or \
            ('from_reload' in st.session_state and st.session_state.from_reload):

            title_filtered_results = data[data['title'].str.contains(search_query, case=False)] 
            pinecone_results = get_pinecone_results(search_query, model, pinecone_index)
            asin_ids = [res['metadata']['asin'] for res in pinecone_results]
            descriptions = data[data['asin'].isin(asin_ids)]['description_str'].tolist()
            cohere_results = rerank_with_cohere(search_query, descriptions, co_client)
            top_results = [asin_ids[res.index] for res in cohere_results]
            pinecone_cohere_results = data[data['asin'].isin(top_results)]
            final_results = pd.concat([title_filtered_results, pinecone_cohere_results])
            final_results = final_results.drop_duplicates(subset=['asin'])
            st.session_state.filtered_products_df = final_results

        else:
            st.session_state.filtered_products_df = data.sample(10, random_state=random.randint(0,100)) # Sample random 10 products
            st.session_state.popped = False  # Reset popped state

    view_products(st.session_state.filtered_products_df)
    st.session_state.popped = True  # Set popped state to True after displaying products
    st.session_state.from_reload = False


def get_all_tabular_categories(data):
    distinct_categories = data['Category'].unique()
    distinct_brands = data['brand'].dropna().sort_values().unique()

    st.session_state.distinct_categories = distinct_categories
    st.session_state.distinct_brands = distinct_brands


def main():
    st.set_page_config(page_title='GameSpace', layout='wide', page_icon="console.png")

    
    data = load_data()
    if 'page' not in st.session_state:
        st.session_state.page = 'search'
    
    if ('query_for_sorting' not in st.session_state) or st.session_state.query_for_sorting == '':
        get_all_tabular_categories(data)

    # SIDEBAR -- disable submit button on product 'view' page
    with st.sidebar:
        with st.expander("How to Use:", expanded=True):
            st.markdown('''* Enter a query and hover over the :grey_question: icon to see relevant reviews\n* View an item to chat with reviews and product specifications\n\n''')
        st.write('\n\n\n')
    
    with st.sidebar.form(key='filter_form'):
        st.markdown('# Search Box :mag:')
        st.markdown('\n\n\n\n')

        click_disabled = False
        if 'page' in st.session_state and st.session_state.page == 'view':
            click_disabled = True
        if ('query_for_sorting' not in st.session_state) or st.session_state.query_for_sorting == '':
            st.text_input("Enter your query:", key='query_for_sorting', disabled=click_disabled)

        else:
            st.text_input("Enter your query:", key='query_for_sorting', value=st.session_state.query_for_sorting, disabled=click_disabled)

        
        

        st.markdown('\n\n\n')
        st.form_submit_button(label='Search', on_click=recommend_games, disabled=click_disabled)

    if st.session_state.page == 'search' and ('product' not in st.session_state):
        st.session_state.from_reload = True
        recommend_games()

    elif st.session_state.page == 'view':
        if 'product' in st.session_state and st.session_state['product'] in data['asin'].values:
            view_product_details(st.session_state['product'], data)

if __name__ == "__main__":
    main()
