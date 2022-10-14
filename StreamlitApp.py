import re

from nltk import ngrams, word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import pysentiment2 as ps
from transformers import pipeline

import spacy
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
import os

SIA = SentimentIntensityAnalyzer()

sp = spacy.load("en_core_web_sm")


def textprocessing(text):
    from nltk import pos_tag, WordNetLemmatizer, word_tokenize
    # tokens = list(gensim.utils.tokenize(text, deacc=True))
    # Remove punctuation
    regex = re.findall("[a-zA-Z'äöü]+", text)
    Text_pre = ""
    for token in regex:
        Text_pre += token.lower() + " "

    Text_pros = word_tokenize(Text_pre)
    tex_new = []
    wnl = WordNetLemmatizer()
    for word, tag in pos_tag(Text_pros):
        if tag.startswith("NN"):
            tex_new.append(wnl.lemmatize(word, pos='n'))
        elif tag.startswith('VB'):
            tex_new.append(wnl.lemmatize(word, pos='v'))
        elif tag.startswith('JJ'):
            tex_new.append(wnl.lemmatize(word, pos='a'))
        else:
            tex_new.append(word)

    stopWords = set(stopwords.words('english'))

    wordsFiltered = ""
    for w in tex_new:
        if w not in stopWords:
            wordsFiltered += w + " "
    return wordsFiltered


def ner(text1):
    sen = sp(str(text1))
    text_no_namedentities = ""
    for token in sen:
        if not token.ent_type:
            text_no_namedentities += token.text
            if token.whitespace_:
                text_no_namedentities += " "
    return text_no_namedentities


def sentiment(text2):  ##has to be after ner

    absolute_path = os.path.dirname(__file__)
    relative_path = "FI_CW.csv"
    full_path = os.path.join(absolute_path, relative_path)
    fi = pd.read_csv(full_path)
    pos_dict = fi[:1050]
    neg_dict = fi.tail(1050)
    neg_dict.reset_index(inplace=True)

    pos_list = pos_dict['Features'].tolist()
    neg_list = neg_dict['Features'].tolist()

    a = text2.split()
    token = word_tokenize(text2)
    bigram = list(ngrams(token, 2))
    pos = []
    neg = []
    pos_score = []
    neg_score = []

    for word in a:
        if word in pos_list:
            index = pos_list.index(word)
            score = pos_dict['mean_fi'][index]
            pos.append(word)
            pos_score.append(score)

        elif word in neg_list:
            neg.append(word)
            index = neg_list.index(word)
            score = neg_dict['mean_fi'][index]
            neg_score.append(score)

    # Look for bigrams
    for bi in bigram:
        cand_bi = ' '.join(list(bi))
        if cand_bi in pos_list:
            pos.append(cand_bi)
            index = pos_list.index(cand_bi)
            score = pos_dict['mean_fi'][index]
            pos_score.append(score)
        elif cand_bi in neg_list:
            neg.append(cand_bi)
            index = neg_list.index(cand_bi)
            score = neg_dict['mean_fi'][index]
            neg_score.append(score)

    try:
        ni = len(neg) / len(a)
        # Neg.clear()
        Neg = ni
    except ZeroDivisionError:
        # print('Stop Dif')
        Neg = 0

    try:
        pi = len(pos) / len(a)
        # Pos.clear()
        Pos = pi
    except ZeroDivisionError:
        # print('Stop Dif')
        Pos = 0

    try:
        fi_sum = (sum(pos_score) + sum(neg_score)) / len(a)
        FI_Score = fi_sum
    except ZeroDivisionError:
        # print('Stop Dif')
        FI_Score = 0

    return FI_Score, Pos, Neg


def results(text):
    if checkbox_our:
        st.write("Results for Our Dictionary")
        nlist = textprocessing(text)
        necList = ner(nlist)
        polarityScore = sentiment(necList)
        col1, col2, col3 = st.columns(3)
        col3.metric("FI_Score", round(polarityScore[0] * 10000) / 10000)
        col1.metric("Pos Score", round(polarityScore[1] * 10000) / 10000)
        col2.metric("Neg Score", round(polarityScore[2] * 10000) / 10000)

    if checkbox_harvard:
        st.write("Results for Harvard Dictionary")
        hiv4 = ps.HIV4()

        def harvard_dict(text1):
            tokens = hiv4.tokenize(text1)
            res = hiv4.get_score(tokens)
            regex = re.findall("[0-9.+-]+", str(res))
            Text_pre = []
            for token in regex:
                t = round(float(token) * 10000) / 10000
                Text_pre.append(t)
            return Text_pre

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Positive", str(harvard_dict(text)[0]))
        col2.metric("Negative", str(harvard_dict(text)[1]))
        col3.metric("Polarity", str(harvard_dict(text)[2]))
        col4.metric("Subjectivity", str(harvard_dict(text)[3]))

    if checkbox_VADER:
        st.write("Results for VADER")
        SIA = SentimentIntensityAnalyzer()
        regex = re.findall("[0-9.+-]+", str(SIA.polarity_scores(text)))
        Text_pre = []
        for token in regex:
            t = round(float(token) * 10000) / 10000
            Text_pre.append(t)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Positive", str(Text_pre[2]))
        col2.metric("Negative", str(Text_pre[0]))
        col3.metric("Neutral", str(Text_pre[1]))
        col4.metric("Compound", str(Text_pre[3]))

    if checkbox_BERT:
        st.write("Results for BERT Model")
        classifier = pipeline("text-classification", model="j-hartmann/sentiment-roberta-large-english-3-classes",
                              return_all_scores=True)
        regex = re.findall("[0-9.]+", str(classifier(text[:512])))
        Text_pre = []
        for token in regex:
            t = round(float(token) * 10000) / 10000
            Text_pre.append(t)

        col1, col2, col3 = st.columns(3)
        col1.metric("Positive", str(Text_pre[2]))
        col2.metric("Negative", str(Text_pre[0]))
        col3.metric("Neutral", str(Text_pre[1]))


def getFatalities(country, year, month):
    absolute_path = os.path.dirname(__file__)
    relative_path = "cw_ts_bert_preprocessed.csv"
    full_path = os.path.join(absolute_path, relative_path)
    fi = pd.read_csv(full_path, sep=';')
    fil = (fi['iso3'] == country)
    fil1 = (fi['year'].astype(str) == str(year))
    fil2 = (fi['month'].astype(str) == str(month))
    noFatalities = True
    for x in range(0, len(fi.index)):
        if fil[x] and fil1[x] and fil2[x]:
            st.metric("Fatalities", str(fi['fatalities'][x]))
            noFatalities = False
            break
    if noFatalities:
        st.write("No fatalities published during this month")

st.set_page_config(
    page_title="Conflict Intensity",
    page_icon="💥",
)
sub = st.container()

with sub:
    st.title("Conflict Intensity")
    st.write(
        "This is a specialized NLP tool for conflict specific text analysis. In order to use this tool you must paste your article into the field below and press submit.")

option = st.selectbox("Choose the kind of text you want to analyse", ("Own text", "Venezuela, June 2004", "COD, December 2007", "Bangladesh, December 2008", "Guinea, February 2010", "Belgium, March 2016"))

# 2003.08-2021.12
st.write("")

if option == "Own text":
    st.write("Select Country, Year and Date of your text for comparison with fatality count")
    c1, c2, c3 = st.columns(3)
    with c1:
        countryOption = st.selectbox("Country", (
            "AFG", "AGO", "ALB", "ARE", "ARM", "AZE", "BDI", "BEL", "BEN", "BFA", "BGD", "BHR", "BIH", "BOL", "BRA",
            "CAF",
            "CHN", "CIV", "CMR", "COD", "COG", "COL", "COM", "DEU", "DJI", "DZA", "ECU", "EGY", "ERI", "ESP", "ETH",
            "FRA",
            "GBR", "GEO", "GHA", "GIN", "GMB", "GNB", "GTM", "GUY", "HND", "HRV", "HTI", "IDN", "IND", "IRN", "IRG",
            "ISR",
            "JAM", "JOR", "KEN", "KGZ", "KHM", "KWT", "LBN", "LBR", "LBY", "LKA", "LSO", "MAR", "MDA", "MDG", "MEX",
            "MKD",
            "MLI", "MMR", "MOZ", "MRT", "MYS", "NER", "NGA", "NIC", "NPL", "PAK", "PER", "PHL", "PNG", "PRY", "QAT",
            "RUS",
            "RWA", "SAU", "SDN", "SEN", "SLB", "SLE", "SLV", "SOM", "SRB", "SSD", "SWZ", "SYR", "TCD", "TGO", "THA",
            "TJK",
            "TUN", "TZA", "UGA", "UKR", "USA", "UZB", "VEN", "YEM", "ZAF", "ZMB", "ZWE"))

    with c2:
        yearOption = st.selectbox("Year", (
            "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015",
            "2016",
            "2017", "2018", "2019", "2020", "2021"))

    with c3:
        monthOption = st.selectbox("Month", (
            "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"))
    # vielleicht hier submit button einfügen
    doc = st.text_area(
        "Paste your text below",
        height=510,
    )

if option == "Venezuela, June 2004":
    text="President Hugo Chávez stepped back from brink of constitutional crisis by accepting recall referendum - set for 15 August - after opposition gathered 2.54 million signatures, surpassing 2.43 million (20% of electorate) required by constitution. Decision avoids direct confrontation with popular opposition; followed highly publicised talks between Chávez and OAS Secretary General Cesar Gaviria and former U.S. president Jimmy Carter. To win referendum opponents must match 3.76 million votes Chávez received in 2000 election. Opposition still concerned government may try to manipulate election process. Should Chávez lose recall before 19 August (completion of 4th year of 6-year term), presidential elections would be held within month. After 19 August, Chávez’s vice president, José Vicente Rangel, would serve remainder of Chávez’s term."
    doc = st.text_area(
        "This is a text from Venezuela, June 2004",
        text,
        height=510,
    )
    countryOption = "VEN"
    yearOption = 2004
    monthOption = 6

if option == "COD, December 2007":
    text="Heavy ﬁghting continued in east throughout month, yet government- sponsored peace conference due 6 January. Intentions of participants unclear and ceaseﬁre not yet observed, but conference offers opportunity to move towards peace. Government launched offensive against forces under Laurent Nkunda in North Kivu with MONUC logistical support, 3 December. Army suffered signiﬁcant defeat to rebels at Mushake, 11 December. Nkunda declared unilateral ceaseﬁre 24 December ahead of 27 December (postponed to 6 January) peace conference. Congolese, Rwandan representatives met in Goma 16 December; proposed implementation taskforce for November Nairobi Communiqué to conduct anti-FDLR operations. U.S. pledged to strengthen FARDC in Kivu at 4-5 December Tripartite Plus Joint Commission summit. Kinshasa signed border security deal with Kampala 15 December. Ugandan LRA rebels based in Garamba national park reportedly attacked town of Duru 16 December, causing mass displacement."
    doc = st.text_area(
        "This is a text from the Democratic Republic of Congo, December 2007",
        text,
        height=510,
    )
    countryOption="COD"
    yearOption=2007
    monthOption=12

if option == "Bangladesh, December 2008":
    text="Largely peaceful 29 Dec polls yielded decisive victory for Awami League (AL), with 230 of 300 seats amid 70% turnout. 2-year state of emergency lifted 17 Dec in advance of polls. Full campaigning began 12 Dec after caretaker govt lifted restrictions on rallies. BNP, which won only 29 of 300 parliamentary seats, initially announced would protest some irregularities in polls, but 1 Jan accepted defeat. AL-led alliance to hold commanding majority in parliament with 262 seats; BNP MPs yet to take oath of office but indicate they will work with new govt even as ongoing post-poll violence between AL and BNP supporters killed 4. AL head Sheikh Hasina and govt due to be sworn in 6 Jan. Supreme Court 19 Dec ruled jailed candidates may contest polls."
    doc = st.text_area(
        "This is a text from Bangladesh, December 2008",
        text,
        height=510,
    )
    countryOption = "BGD"
    yearOption =2008
    monthOption =12

if option == "Guinea, February 2010":
    text="PM Doré 15 Feb appointed 34-person interim govt comprising opposition, trade unionists and former junta members. Electoral commission 21 Feb proposed presidential polls for 27 June, welcomed by ECOWAS and Contact Group for Guinea 22 Feb. Doré called for elections support from donors. Govt inquiry into 28 Sept massacre issued report 2 Feb alleging Aboubacar “Toumba” Diakité, presidential guard member accused of shooting ex-junta leader Camara in Dec, was solely responsible for Sept killings. Diakité 5 Feb announced he had been following orders and was willing to face justice but not in Guinean court. ICC assessment mission arrived mid-month, 19 Feb announced Sept killings could amount to “crimes against humanity” and pledged to press forward with preliminary investigation. During mission, Doré stated Guinea’s judiciary was unfit to undertake prosecutions properly."
    doc = st.text_area(
        "This is a text from Guinea, February 2010",
        text,
        height=510,
    )
    countryOption = "GIN"
    yearOption =2010
    monthOption =2

if option == "Belgium, March 2016":
    text="32 people killed and over 300 injured by two suicide bomb attacks at Brussels airport and on metro in Brussels 22 March; both attacks claimed by Islamic State (IS). Three attackers also killed in explosions, one suspect remains at large, several arrested over suspected involvement in attack and with Europe-wide network linked to IS in Syria. Attacks followed capture in Brussels 18 March of Salah Abdeslam, suspect in Nov 2015 Paris attacks, and 15 March raid, also in Brussels, in which another Paris attacks suspect killed."
    doc = st.text_area(
        "This is a text from Belgium, March 2016",
        text,
        height=510,
    )
    countryOption ="BEL"
    yearOption =2016
    monthOption =3


with st.form(key="Type of Dictionary"):
    st.write("Choose one or more:")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        checkbox_our = st.checkbox("Our Dictionary")
    with c2:
        checkbox_harvard = st.checkbox("Harvard IV-4")
    with c3:
        checkbox_BERT = st.checkbox("BERT Model", help="Might take a few seconds")
    with c4:
        checkbox_VADER = st.checkbox("VADER")

    submit_button_2 = st.form_submit_button(label="Get Data ")

if submit_button_2:
    results(doc)

    getFatalities(countryOption, yearOption, monthOption)

    with st.expander("Value Explanation"):
        st.header("Our Dictionary")
        st.write(
            "The three output scores (Pos, Neg and FI_Score) measure the overall conflict intensity (in terms of fatalities) of a given document.")
        st.write(
            "The Pos and Neg scores indicate how many words are positively or negaticely correlated with fatalities, indicating a higher/lower conflict intensity. Both scores are divided by the amount of words of the pre-processed document and can therefore be understood as the relative word frequency of 'positive' and 'negative' words. ")
        st.write(
            "Instead of relying on simple word counts, the FI Score takes into consideration the relative importance of each word. It is the sum of the feature importance of 'positiv' and 'negative' words divided by document length.")
        st.markdown("")
        st.header("Harvard Dictionary")
        st.write("some desciption")
        st.markdown("")
        st.header("BERT Model")
        st.write("some desciption")
