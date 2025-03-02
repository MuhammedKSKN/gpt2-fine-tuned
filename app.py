import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Hugging Face model ismi
model_name = "mkesks/my-gpt2-ft"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

text_generator = load_model()

# Streamlit arayüzü
st.title("💬 GPT-2 Fine-Tuned Model")
st.write("Aşağıya bir metin girin ve modelin tamamlamasını görün!")

user_input = st.text_area("Metni buraya gir:", "Merhaba, nasılsın?")

if st.button("Metni Tamamla"):
    if user_input:
        with st.spinner("Model çalışıyor..."):
            result = text_generator(user_input, max_length=100, num_return_sequences=1)
            st.success(result[0]["generated_text"])
    else:
        st.warning("Lütfen bir giriş metni girin!")
