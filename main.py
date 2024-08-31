import streamlit as st
import textdistance
import time
import generator
import pandas as pd
dataset = pd.read_csv(".\embeddings\sectionsdataset_withoutillustrations - Sheet1.xls.csv")

# Function to find the most similar word in a list
def find_similar_word(input_word, word_list):
    similarities = [(word, textdistance.jaccard(input_word, word)) for word in word_list]
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[0][0]

def typewriter_text(message,container):
    typed_message = ""
    for char in message:
        typed_message += char
        container.write(typed_message)
        time.sleep(0.025)

def writefromlist(result,parent):
    with parent:
        for i in result:
            newcontainer = st.empty()
            sectionum = str(dataset.loc[dataset['Section']==i[0],'Number']).split(" ")[4].replace("\nName:","")
            print(sectionum)
            typewriter_text("o "+i[0]+ " - Section Number: "+sectionum,newcontainer) 
            newcontainer.empty()
            st.button("o "+i[0]+" - Section Number: "+sectionum)
            desc = dataset.loc[dataset['Section']==i[0],'Description']
            desccont = st.empty()
            typewriter_text(desc,desccont)
               

        
# Main function to create the Streamlit app
def main():
    # Setting page configuration
    st.set_page_config(
        page_title="LEGAL ADVISORY SYSTEM",
        page_icon=".\icon\octicon--law-16.svg",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

   

    st.markdown('<style>' + open('./main.css').read() + '</style>', unsafe_allow_html=True)


    st.title("Welcome to Legal Assistance System (LAS)")
    st.write("Your First Stop for any LEGAL advise")
    writtenonce=0
    # Input box for user input
    chatbt = st.empty()
    
    response_container = st.container()

    user_input = st.text_input("Enter Response")


    
    # Button to trigger the correction
    if st.button("Enter"):
        instance  = generator.suggestor()
        writefromlist(instance.calculate_similarity(user_input)[:5],response_container)
    if(writtenonce==0):
        typewriter_text("Hii !!!!!! I am LAS short for Legal Advisory System. State the incident or problem you're facing. I will help you with a solution.",chatbt)
        writtenonce = 1
    else:
        typewriter_text("Hope this helped. Are there any other queries ?",chatbt)
    
    

if __name__ == "__main__":
    main()
