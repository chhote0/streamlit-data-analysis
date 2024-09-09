import streamlit as st
from langchain.llms import Llama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
import pandas as pd

model = OllamaLLM(model="llama3")

# Prompt şablonunu oluşturun
prompt_template = PromptTemplate(
    input_variables=["csv_data", "user_question"],
    template="CSV dosyasındaki verilere göre şu soruya cevap ver: {user_question}\nCSV verileri:\n{csv_data}"
)

def process_csv(uploaded_file):
    # CSV dosyasını okuyun
    df = pd.read_csv(uploaded_file)

    # CSV verilerini bir metin dizesine dönüştürün (burada daha gelişmiş bir dönüşüm yöntemi kullanabilirsiniz)
    csv_text = df.to_string()

    # Kullanıcıdan soru alın
    user_question = st.text_input("Sorunuzu girin:")

    # Prompt'ı oluşturun ve LLM ile çalıştırın
    prompt = prompt_template.format(csv_data=csv_text, user_question=user_question)
    chain = LLMChain(llm=model, prompt=prompt)
    response = chain.run()

    # Cevabı gösterin
    st.text_area("Cevap:", value=response, height=200)

# Uygulama başlığı
st.title("CSV Soru-Cevap Botu")

# CSV dosyası yükleme
uploaded_file = st.sidebar.file_uploader("CSV dosyası seçin", type="csv")

# Dosya yüklendiğinde işlemi başlat
if uploaded_file is not None:
    process_csv(uploaded_file)




# import streamlit as st
# from langchain.document_loaders.csv_loader import CSVLoader
# import tempfile
# from utils import get_model_response




# # Main app
# def main():
#     st.title("Chat with CSV using Llama")


#     # File Uploader
#     uploaded_file = st.sidebar._file_uploader("Choose a CSV file", type="csv")

#     # Fetching path of the uploaded file
#     if uploaded_file is not None:
#         # use tempfile because CSVLoader only accepts a file_path
#         with tempfile. NamedTemporaryFile(delete=False) as tmp_file:
#             tmp_file.write(uploaded_file.getvalue())
#             tmp_file_path = tmp_file.name
            
#             # Initializing CSV_Loader
#             csv_loader = CSVLoader (file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ', '})

#             # Load data into csv loader
#             data = csv_loader.load()

#             # Initialize chat Interface
#             user_input = st.text_input("Your Message:")
#             print(user_input)

#             if user_input:
#                 get_model_response(data,user_input)
#                 response = "response"
#                 st.write(response)



# if __name__ == "__main__":
#     main()