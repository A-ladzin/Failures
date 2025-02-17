import pickle
import json
import streamlit as st

from backend.src.core.data import preprocess, predict, explain_prediction
from backend.src.core.definitions import DATA_DIR, MODEL_DIR

from backend.src.models.model_coupled import CoupledBert
from backend.src.models.nli import NLIModel
from transformers import BertModel, AutoTokenizer
from backend.src.models.utils import load_model

st.set_page_config(
    page_title="DLS | Trends Indicator",
    page_icon="💬"
)

def schema_state_init()-> None:
    if "model" not in st.session_state:


        bert_tokenizer = AutoTokenizer.from_pretrained("ai-forever/sbert_large_nlu_ru")
        bert_tokenizer.add_tokens(['[QUERY]','[DESCRIPTION]','[NUM]'])
        bert = BertModel.from_pretrained("10-8nli")
        bert.eval()
            
        nli_model = NLIModel(bert,bert_tokenizer,embedding_dim=512,k=50,mask_prob=0)
        load_model(nli_model,"10-8mean_sbert8___/checkpoint-404")
        bert.requires_grad_(False)

        model = CoupledBert(nli_model,bert_tokenizer,50,50)
        model.prepare_description_tokens()
        

        st.session_state["model"] = model


    if "mapping" not in st.session_state:
        with open(DATA_DIR / 'mapping_backend.json', 'r') as f:
            st.session_state["mapping"] = json.load(f)

def clear_text() -> None:
    st.session_state["text_input_area"] = ""

def main()-> None:
    
    schema_state_init()

    st.header(f"Классификация Пользовательского контента", divider='rainbow')     
    
    st.markdown("<b>Описание:</b><br>\
                В рамках данного примера, мы решаем задачу множественной классификации для определения всех классов, к которым можно отнести отзыв Пользователя о Сервисе.\
                Множественная классификация отличается от многоклассовой тем, что экземпляр данных может одновременно относиться сразу к нескольким классам.\
                На вход модели мы подаём только комментарий Пользователя и пытаемся определить для него 50 различных меток классов, к которым он может относиться.", unsafe_allow_html = True)

    review = st.text_area(label = "Введите текст отзыва:", key = "text_input_area", value = "")

    col_1, col_2 = st.columns(2)

    col_1.button("Очистить", use_container_width = True, on_click=clear_text)
    if col_2.button("Отправить", type = "primary", use_container_width = True):
        if len(review) == 0:
            st.write(":red[Длина сообщения не может быть нулевой!]")
        else:
            data = preprocess(review)
            prediction = predict(data, st.session_state["model"])
            trends_list = explain_prediction(st.session_state["mapping"], prediction)
            if len(trends_list) > 0:
                st.write("<br><br>".join([f"<b>Name:</b> {trend[0]}<br><b>Description:</b><br>{trend[1]}" for trend in trends_list]), unsafe_allow_html = True)
            else:
                st.write(":red[Не удалось определить тренды:(]")
    
    return None

if __name__ == "__main__":
    main()