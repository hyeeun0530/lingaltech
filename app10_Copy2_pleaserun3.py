import streamlit as st
import os
import pandas as pd
import time
import re

# --- IMPORTS ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# TOGETHER AI IMPORTS
try:
    from langchain_together import ChatTogether
except ImportError:
    st.error("❌ 'langchain-together' library not found. Please run: pip install -r requirements.txt")
    st.stop()

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="AI Labor Law (Model Answer)", page_icon="⚖️", layout="wide")

# FILES
CSV_FILE = "04. 법률팀 라벨링 매뉴얼.csv"

# CREDENTIALS (PRE-FILLED)
DEFAULT_API_KEY = "e56ad0c3f4a4cb169e4d916d5a474488eb17fe444c0910b3512d8f9beffae275"
DEFAULT_MODEL = "lingaltech810/Meta-Llama-3-8B-Instruct-63c67d9f-93eddf60"

# STATE
if "step" not in st.session_state: st.session_state.step = "A"
if "user_profile" not in st.session_state: st.session_state.user_profile = {}
if "contract_text" not in st.session_state: st.session_state.contract_text = ""
if "selected_example" not in st.session_state: st.session_state.selected_example = None
if "original_contract_text" not in st.session_state: st.session_state.original_contract_text = ""

# --- SHARED PROGRESS STEPS (IDENTICAL FOR FAKE & REAL) ---
PROGRESS_STEPS = [
    "1. 근로자 프로필 분석",
    "2. 계약서 조항 파싱",
    "3. 법률 데이터베이스 검색",
    "4. 위험 조항 식별",
    "5. 법적 근거 매칭",
    "6. 개선안 도출",
    "7. 최종 분석 완료"
]

# --- 2. MODEL ANSWERS ---
# --- 2. MODEL ANSWERS (EXACT FROM PDF, ANONYMIZED) ---
MODEL_ANSWERS = {
    "example_pregnant": {
        "profile_display": "내국인, 여성, 임신 중, 만 30세, 비장애인, 일반근로시간 유형",
        "summary": """귀하의 근로계약서에서 임신 중 여성 근로자의 근로조건 보호와 관련하여 일부 조항에서 법적 주의 필요 항목이 식별되었습니다.

전체적으로는 다음과 같습니다. 총 11개 조항 중 저위험 7개, 중위험 1개, 고위험 3개입니다.

특히 근로시간, 임금, 유해·위험 업무 제한 관련 항목은 「근로기준법」제65조~제74조 및 「남녀고용평등과 일·가정 양립 지원에 관한 법률」제19조를 반드시 확인해야 합니다.""",
        "details": """1. 근로개시일 (저위험)

입력: 2025년 1월 18일

사유: 근로 시작일이 명확히 기재되어 있어 법적 기준에 부합하며 안전합니다.

근거: 근로기준법 제17조(근로조건의 명시)

2. 근무장소 (저위험)

입력: 갑 소유의 A 선박

사유: 근무 장소가 명확히 기재되어 있어 법적 기준에 부합하며 안전합니다.

근거: 근로기준법 제17조

3. 업무의 내용 (고위험)

입력: 항만하역업

사유: 임신 중 여성은 신체를 심하게 굽히거나 펴야 하는 업무에 종사할 수 없습니다.

근거: 근로기준법 제65조(사용 금지)

개선: 선상 작업 및 하역 업무 대신 비육체적 업무로 전환하십시오.

4. 소정근로시간 (저위험)

입력: 10:00~13:30 (휴게 30분 포함, 주 12시간)

사유: 근로시간이 단시간으로, 임신 중 근로자 보호 기준에 부합하며 안전합니다.

근거: 근로기준법 제50조, 제71조

5. 근무일·휴일 (고위험)

입력: 미기재

사유: 근무일 및 주휴일은 필수적으로 명시해야 합니다. 누락 시 유급휴일 산정이 불가능합니다.

근거: 근로기준법 제55조(휴일)

개선: 주 5일 근무, 주휴일(예: 일요일) 등을 명시하십시오.

6. 임금 (고위험)

입력: 시급 11,000원, 매월 20일 지급, 계좌이체

사유: 최저임금 이상이지만, 출산휴가 기간 중 급여 지급 규정이 누락되었습니다.

근거: 근로기준법 제74조(임산부 보호), 고용보험법 제70조(출산전후휴가급여)

개선: 출산휴가(90일, 다태아 120일) 및 급여 지급 주체를 명시하십시오.

7. 연차유급휴가 (저위험)

입력: 근로기준법에 따라 부여

사유: 법정 연차 기준에 부합하며 안전합니다.

근거: 근로기준법 제60조

8. 사회보험 적용여부 (중위험)

입력: 산재보험, 건강보험 적용

사유: 월 60시간 미만 근로자는 국민연금·고용보험 적용 대상에서 제외될 수 있습니다.

근거: 국민연금법 제6조, 고용보험법 제10조

개선: 각 공단을 통해 실제 적용 여부를 확인하십시오.

9. 근로계약서 교부 (저위험)

입력: 계약 체결 시 교부

사유: 교부 의무 명시되어 있어 법적 기준에 부합하며 안전합니다.

근거: 근로기준법 제17조

10. 근로계약·취업규칙 성실 이행 (저위험)

입력: 성실 이행 의무 명시

사유: 근로자·사용자 간 상호 성실 이행이 규정되어 있어 안전합니다.

근거: 근로기준법 제5조

11. 그 밖의 사항 (저위험)

입력: 근로관계법령에 따름

사유: 법령 준수 문구가 포함되어 있어 법적 기준에 부합하며 안전합니다.

근거: 근로기준법 제15조""",
        "legal_guide": """1. 임금 항목

참조 법령: 「최저임금법」 제6조, 「근로기준법」 제43조, 「고용보험법」 제70조

요지: 사용자는 근로자에게 고용노동부 장관이 고시하는 최저임금 이상을 지급해야 하며(최저임금법 제6조), 임금은 통화로, 직접, 전액을, 매월 1회 이상 일정한 날짜에 지급해야 합니다(근로기준법 제43조). 근로기준법 제74조에 따라 출산전후휴가의 최초 60일은 유급으로 보장 받습니다. 또한, 육아휴직 이후 1개월~12개월 이내의 여성은 고용보험법 제70조에 따라 육아휴직급여를 신청할 수 있습니다.

2. 근로시간 항목

참조 법령: 「근로기준법」 제50조, 제71조

요지: 1일 근로시간은 8시간, 1주 근로시간은 40시간을 초과할 수 없습니다(제50조). 임신 중 여성 근로자는 시간외근로(연장근로)를 시킬 수 없으며, 야간·휴일근로도 원칙적으로 금지됩니다(제71조).

3. 유해·위험 업무 관련 항목

참조 법령: 「근로기준법」 제65조

요지: 사용자는 임신 중 또는 출산 후 1년이 지나지 않은 여성을 도덕상 또는 보건상 유해·위험한 사업에 사용할 수 없습니다. 항만하역업, 선상 중량물 작업 등은 대표적인 금지 직종으로 분류됩니다.

4. 연차유급휴가 항목

참조 법령: 「근로기준법」 제60조

요지: 사용자는 1년간 80% 이상 출근한 근로자에게 15일 이상의 유급휴가를 부여해야 합니다. 1년 미만 근로자는 1개월 개근 시 1일의 유급휴가를 받을 수 있습니다.

5. 사회보험 항목

참조 법령: 「국민연금법」 제6조, 「고용보험법」 제8조 및 제10조, 「산재보험법」 제5조

요지: 사용자는 근로자를 국민연금, 고용보험, 산재보험에 가입시켜야 합니다. 다만, 월 소정근로시간이 60시간 미만인 경우 국민연금 및 고용보험은 적용 제외 대상이 될 수 있습니다.

6. 출산휴가 및 보호 관련 항목

참조 법령: 「근로기준법」 제74조, 「남녀고용평등과 일·가정 양립 지원에 관한 법률」제19조

요지: 임신 중인 여성은 출산 전후 총 90일(미숙아 100일, 다태아 120일)의 휴가를 받을 수 있습니다(근로기준법 제74조). 출산휴가기간 중 급여는 고용보험에서 지급되며, 사업주는 이를 이유로 불이익을 주어서는 안 됩니다(남녀고용평등법 제19조)."""
    },
    
    "example_executive": {
        "profile_display": "내국인, 남성, 만 35세, 장애 없음, 포괄임금제 근로자",
        "summary": """귀하의 근로계약서 중 근로조건의 명시 항목과 포괄임금제 계약과 관련하여 일부 조항에서 법적 주의 필요 항목이 식별되었습니다.

전체적으로는 다음과 같습니다. 총 11 개 조항 중 저위험 6개, 중위험 3개, 고위험 2개입니다.

근로조건의 명시 항목과 관련하여 「근로기준법」 제17조1항을 확인하시고 임금 항목과 관련하여서는 「근로기준법」 제56조를 반드시 확인하시기 바랍니다.""",
        "details": """1. 근로개시일 (중위험)

입력: 2025년 1월 1일부터
*수습기간은 근로개시일로부터 3개월, 수습기간을 무난히 통과한 이후에는 계약기간을 1년 단위(수습기간 포함)로 하며, 상호 이의가 없을 시에는 1년 단위로 자동연장하기로 한다.

사유: 본 계약은 ‘2025년 1월 1일부터'로 기재되어 있어 기간의 정함이 없는 정규직 근로계약으로 보일 수 있으나, 법적 분쟁시 ‘1년 단위로 자동연장하기로 한다'라는 문구로 인해 ‘기간의 정함이 있는 계약'으로 판단할 소지가 있습니다.

근거: 근로기준법 제 17조

개선: 본 계약은 기간의 정함이 있는 계약으로 볼 소지가 있음을 인지하고 정규직, 비정규직 여부에 대하여 명확히 규정할 수 있도록 사용자와 상의해야 합니다.

2. 근무장소 (중위험)

입력: 본사에서 지정한 장소. 단, 담당업무 및 근무 장소는 회사 사정에 따라 변경할 수 있다.

사유: 근무 장소가 명확히 기재되어 있지 않아 근무지가 임의로 변경될 수 있기에 근로자에게 불리한 조항으로 작용할 수 있습니다.

근거: 근로기준법 제 17조

개선: 근무지가 명확하게 특정될 수 있도록 기재하십시오.

3. 업무의 내용 (중위험)

입력: 부사장

사유: 근로계약서 상 업무의 내용 및 직책에 부사장 등과 같이 임원직의 명칭으로 기재되어 있더라도 근무형태나 법적관계가 근로자에 해당한다면 근로기준법에 따라 근로자로서 보호를 받아야 합니다.

근거: 근로기준법 제17조

개선: 근무형태나 법적관계에 따라 임원직이 아닌, 사용자와 종속관계에 따른 근로자에 해당할 수 있음을 확인하고 이를 명확히 규정할 수 있도록 사용자와 상의해야 합니다.

4. 소정근로시간 (저위험)

입력: 소정근로시간 : 09시 00분 ~ 18 시 00분 (휴게: 12 시 00분 ~ 13 시 00 분) (1일 8 시간, 1주 40시간)

사유: 근로시간이 법정근로시간을 초과하지 않고 휴게시간 또한 적절히 기재되어 있어 법적 기준에 부합하며 안전합니다.

근거: 근로기준법 제17조, 근로기준법 제50조

5. 근무일·휴일 (저위험)

입력: 매주 5일 근무, 주휴일 매주 일요일, 공휴일(대체공휴일 포함)은 근로기준법이 정하는 바에 따르며, 근로자의 날은 유급휴일로 함

사유: 근무일과 휴일이 적절하게 기재되어 있어 법적 기준에 부합하며 안전합니다.

근거: 근로기준법 17조, 근로기준법 55조

6. 임금 (고위험)

입력: 연봉 50,000,000원, 수습기간 중 계약 연봉의 100%를 지급한다. 상여금 없음. 그 밖의 수당 없음. 매월 5일 임금 지급, 계좌 입금

사유: 본 계약서는 포괄임금제를 기반하여 작성되었기에 연장근로, 야간근로, 휴일근로에 따른 법적 수당이 임금 안에 가산되어 있는지 확인해야 합니다.

근거: 근로기준법 제56조

개선: 근로일수 및 시간외근로에 따라 지불될 임금이 각종 법적 수당을 정확히 반영하고 있는지 확인하십시오.

7. 연차유급휴가 (저위험)

입력: 연차유급휴가는 근로기준법에서 정하는 바에 따라 부여함.

사유: 법령 준수 문구가 포함되어 있어 법적 기준에 부합하며 안전합니다.

근거: 근로기준법 제15 조, 근로기준법 제60조

8. 사회보험 적용여부 (저위험)

입력: 4대 사회보험(고용보험, 산재보험, 국민연금, 건강보험) 적용(가입)을 원칙으로 함

사유: 사회보험 적용 원칙을 규정하고 있어 안전합니다.

근거: 국민연금법 제8조

9. 근로계약서 교부 (저위험)

입력: 계약 체결 시 교부

사유: 교부 의무 명시되어 있어 법적 기준에 부합하며 안전합니다.

근거: 근로기준법 제 17조

10. 근로계약·취업규칙 성실 이행 (고위험)

입력: 성실 이행 의무 명시, 자리를 비울 때에는 반드시 상사에게 보고하고 컨펌 하에 움직이도록 한다.

사유: 자리를 비울 때'라는 문구로 인해 모든 이석을 상사의 허가 대상으로 삼을 수 있다는 의미로 해석될 여지가 있으며 이는 근로자의 자율성을 지나치게 제약할 수 있습니다.

근거: 근로기준법 제7조, 근로기준법 제76조

개선: 해당 조항을 삭제할 수 있도록 하십시오.

11. 그 밖의 사항 (저위험)

입력: 근로관계법령에 따름

사유: 법령 준수 문구가 포함되어 있어 법적 기준에 부합하며 안전합니다.

근거: 근로기준법 제 15 조""",
        "legal_guide": """1. 임금 항목

참조 법령: 「최저임금법, 「근로기준법」제43조

요지: 사용자는 최저임금법에 따라 고용노동부장관이 정한 최저임금액 이상의 금액을 임금으로 지급해야 하며, 근로기준법 제43조에 따라 매월 1회 이상 일정한 날짜를 정하여 근로자에게 통화로 직접 지급해야 합니다.

2. 근로 조건 명시 항목

참조 법령: 「근로기준법」제17조

요지: 사용자는 근로기준법 제17조에서 규정하는 근로조건을 계약서에 정확하게 명시하여야 합니다. 해당 사항으로는 임금, 소정근로시간, 휴일, 취업의 장소와 종사하여야 할 업무에 관한 사항 등이 존재합니다. 위 사항들이 기재되어 있지 않거나 중의적으로 해석될 문구로 기재되어 있는 경우 주의하여야 합니다.

3. 근로시간 항목

참조 법령: 「근로기준법」제50조, 제53조

요지: 1주 간의 근로시간은 휴게시간을 제외하고 40시간을 초과할 수 없으며 1일의 근로시간은 휴게시간 1시간을 제외하고 8시간을 초과할 수 없습니다. 근로기준법 제50조 제1항 및 제2항에 따라 근로시간을 산정하는 경우 작업을 위하여 근로자가 사용자의 지휘ㆍ감독 아래에 있는 대기시간 등은 근로시간으로 간주합니다. 또한 근로기준법 제53조에 따라 당사자 간에 합의하면 1주 간에 12시간을 한도로 제50조의 근로시간을 연장할 수 있습니다.

4. 연장•야간•휴일수당 항목 (포괄임금제)

참조 법령: 「근로기준법」제56조

요지: 사용자는 근로기준법 제53조·제59조 및 제69조 단서에서 규정하는 연장근로에 대하여는 통상임금의 100분의 50 이상을 가산하여 근로자에게 지급하여야 합니다. 또한, 사용자는 8시간 이내의 휴일근로에 대해서 통상임금의 100분의 50을 지급해야 하며 8시간을 초과한 휴일근로에 대해서 통상임금의 100분의 100을 지급해야 합니다. 뿐만 아니라, 사용자는 오후 10시부터 다음 날 오전 6시 사이의 근로인 야간근로에 대하여는 통상임금의 100분의 50 이상을 가산하여 근로자에게 지급하여야 합니다. 근로 형태나 업무 성질상 추가근무수당을 정확히 집계하기 어려운 경우에 수당을 급여에 미리 포함하는 포괄임금제로 계약이 체결된 경우 위 수당들이 정확히 반영되어 있는지 정확히 확인해야 합니다.

5. 연차유급휴가 항목

참조 법령: 「근로기준법」제 60 조

요지: 사용자는 1 년간 80% 이상 출근한 근로자에게 15 일 이상의 유급휴가를 부여해야 합니다. 1 년 미만 근로자는 1 개월 개근 시 1 일의 유급휴가를 받을 수 있습니다."""
    },

    "example_cafe": {
        "profile_display": "내국인, 남성, 만19세, 장애 없음, 포괄임금제 근로자",
        "summary": """귀하의 근로계약서에서 근로기준법 제 17조에서 규정하고 있는 근로조건의 명시와 관련하여 주의 필요 항목이 식별되었습니다.

전체적으로는 다음과 같습니다. 총 11개 조항 중 저위험 7개, 고위험 4개입니다.""",
        "details": """1. 근로개시일 (저위험)

입력: 2025년 1월 1일부터 12월 31일까지 (단, 3개월은 수습기간으로 한다.)

사유: 근로 계약 기간과 수습기간이 명시되어 법적 기준에 부합합니다.

근거: 근로기준법 제17조

2. 근무장소 (저위험)

입력: 서울특별시 종로구 A카페

사유: 근무장소가 명확히 기재되어 법적 기준에 부합합니다.

근거: 근로기준법 제17조

3. 업무의 내용 (저위험)

입력: 카페 업무 전반

사유: 업무의 내용이 기재되어 법적 기준에 부합합니다.

근거: 근로기준법 제17조

4. 소정근로시간 (고위험)

입력: 08:30~12:30(휴게시간은 조기퇴근으로 대신한다.)

사유: 근로기준법 제54조에는 '휴게시간은 근로시간 중에 제공되어야 한다'고 규정되어 있습니다. 따라서 '휴게시간을 조기퇴근으로 대체한다'는 조항은 근로자가 보장받아야 할 휴게시간을 보장하지 않는 불법적인 조항에 해당합니다.

근거: 근로기준법 제54조

개선: 휴게시간을 근로시간 내에서 지정하여 계약서에 포함하십시오.

5. 근무일·휴일 (고위험)

입력: 주 4일(월, 화, 목, 금) 근무, 주휴일 없음

사유: 근로기준법 제55조에 따라 1주 동안 소정근로일에 모두 출근한 근로자에게 평균 1회 이상의 유급휴일을 보장해야 합니다. 따라서 '주휴일 없음'이라는 조항은 근로자가 보장받아야 할 유급휴일을 보장하지 않는 불법적인 조항에 해당합니다.

근거: 근로기준법 제55조

개선: 1주 1회 이상의 주휴일을 계약서에 포함하십시오.

6. 임금 (고위험)

입력: 시급 11,000원, 상여금 없음, 그 밖의 수당 없음, 매월 27일 지급, 계좌 입금, 수습기간의 임금은 시급의 80%로 정한다.

사유: 최저임금법 제5조 시행령에 따라 1년 이상의 기간을 정하여 근로계약을 체결하고 수습 중인 근로자에 대해 수습기간이 3개월 이내인 경우, 수습기간 동안의 임금은 최저임금의 100분의 10을 감액할 수 있습니다. 즉, 수습기간 동안의 임금은 최저임금의 90% 이상이어야 함을 의미합니다.

근거: 최저임금법 제5조 시행령

개선: 수습기간의 임금을 시급의 90% 이상으로 조정하십시오.

7. 연차유급휴가 (고위험)

입력: 단시간근무자는 연차유급휴가를 지급하지 않는다.

사유: 근로기준법 제18조 시행령에 따라 주 소정근로시간이 15시간 이상인 단시간근로자는 연차유급휴가를 보장받아야 합니다.

근거: 근로기준법 제18조 시행령

개선: 연차유급휴가를 부여하도록 조항을 수정하십시오.

8. 사회보험 적용여부 (저위험)

입력: 4대 사회보험(고용보험, 산재보험, 국민연금, 건강보험) 적용(가입)을 원칙으로 함

사유: 사회보험 적용 원칙을 규정하고 있어 안전합니다.

근거: 국민연금법 제8조

9. 근로계약서 교부 (저위험)

입력: 사업주는 근로계약을 체결함과 동시에 본 계약서를 사본하여 근로자의 교부요구와 관계없이 근로자에게 교부함

사유: 근로계약서 교부 의무를 명시하여 안전합니다.

근거: 근로기준법 제17조

10. 근로계약·취업규칙 성실 이행 (저위험)

입력: 사업주와 근로자는 각자가 근로계약, 취업규칙, 단체협약을 지키고 성실하게 이행하여야 함

사유: 근로 조건의 준수 의무를 명시하여 안전합니다.

근거: 근로기준법 제5조

11. 그 밖의 사항 (저위험)

입력: 없음

사유: 특이사항 없음

근거: 해당없음""",
        "legal_guide": """1. 임금 항목

참조 법령: 「최저임금법」 제6조, 「근로기준법」 제43조, 「고용보험법」 제70조

요지: 사용자는 근로자에게 고용노동부 장관이 고시하는 최저임금 이상을 지급해야 하며(최저임금법 제6조), 임금은 통화로, 직접, 전액을, 매월 1회 이상 일정한 날짜에 지급해야 합니다(근로기준법 제43조). 근로기준법 제74조에 따라 출산전후휴가의 최초 60일은 유급으로 보장 받습니다. 또한, 육아휴직 이후 1개월~12개월 이내의 여성은 고용보험법 제70조에 따라 육아휴직급여를 신청할 수 있습니다.

2. 근로시간 항목

참조 법령: 「근로기준법」 제50조, 제71조

요지: 1일 근로시간은 8시간, 1주 근로시간은 40시간을 초과할 수 없습니다(제50조). 임신 중 여성 근로자는 시간외근로(연장근로)를 시킬 수 없으며, 야간·휴일근로도 원칙적으로 금지됩니다(제71조).

3. 연차유급휴가 항목

참조 법령: 「근로기준법」 제60조

요지: 사용자는 1년간 80% 이상 출근한 근로자에게 15일 이상의 유급휴가를 부여해야 합니다. 1년 미만 근로자는 1개월 개근 시 1일의 유급휴가를 받을 수 있습니다.

4. 사회보험 항목

참조 법령: 「국민연금법」 제6조, 「고용보험법」 제8조 및 제10조, 「산재보험법」 제5조

요지: 사용자는 근로자를 국민연금, 고용보험, 산재보험에 가입시켜야 합니다. 다만, 월 소정근로시간이 60시간 미만인 경우 국민연금 및 고용보험은 적용 제외 대상이 될 수 있습니다."""
    },

    "example_53yo": {
        "profile_display": "내국인, 여성, 만53세, 장애 없음, 특별 근로시간 유형 해당 안 됨",
        "summary": """귀하의 근로계약서에서 총 11개 조항 중 저위험 6개, 중위험 2개, 고위험 3개가 확인되었습니다.

특히 근로시간, 근무일·휴일, 임금 항목과 관련하여 근로기준법 제50조, 제55조, 최저임금법 제6조를 확인하시기 바랍니다.""",
        "details": """1. 근로개시일 (저위험)

입력: 2025년 1월 1일

사유: 근로 시작일이 명확히 기재되어 법적 기준에 적합합니다.

근거: 근로기준법 제17조

2. 근무장소 (중위험)

입력: 회사 필요에 따라 전국 사업장 어디든 발령 가능

사유: 포괄적 표현으로 인해 전국 전근 강제 해석이 가능합니다.

근거: 근로기준법 제17조

개선: ‘주된 근무지 + 전근 시 사전협의' 식으로 범위를 축소하여 명시하십시오.

3. 업무의 내용 (중위험)

입력: 회사 지시하는 모든 업무 일체

사유: 포괄적 표현으로 인해 정확한 근로 내용 파악이 어렵습니다.

근거: 근로기준법 제17조

개선: 담당할 직무 범위를 구체적으로 명시하십시오.

4. 소정근로시간 (고위험)

입력: 08:00~18:00(휴게 1시간), 1일 9시간·주 45시간

사유: 법정 근로시간(1일 8시간, 주 40시간)을 초과하였습니다.

근거: 근로기준법 제50조

개선: 근로시간을 법정 근로시간에 부합하도록 조정하십시오.

5. 근무일·휴일 (고위험)

입력: 주 5일 근무 / 주휴일: 일요일(무급)

사유: 주휴일은 유급이 원칙입니다.

근거: 근로기준법 제55조

개선: 주휴일을 ‘유급휴일'로 조정하십시오.

6. 임금 (고위험)

입력: 월 2,000,000원

사유: 최저임금에 미달합니다.

근거: 최저임금법 제6조

개선: 2025년 최저임금에 부합하도록 조정하십시오.

7. 연차유급휴가 (저위험)

입력: 근로기준법에 따라 부여

사유: 법정 연차 기준에 부합하며 안전합니다.

근거: 근로기준법 제 60조

8. 사회보험 적용여부 (저위험)

입력: 4대 사회보험(고용보험, 산재보험, 국민연금, 건강보험) 적용(가입)을 원칙으로 함

사유: 사회보험을 적용받고 있어 안전합니다.

근거: 국민연금법 제8조

9. 근로계약서 교부 (저위험)

입력: 계약 체결 시 교부

사유: 교부 의무 명시되어 있어 법적 기준에 부합하며 안전합니다.

근거: 근로기준법 제17조

10. 근로계약·취업규칙 성실 이행 (저위험)

입력: 성실 이행 의무 명시

사유: 근로자·사용자 간 상호 성실 이행이 규정되어 있어 안전합니다.

근거: 근로기준법 제5조

11. 그 밖의 사항 (저위험)

입력: 근로관계법령에 따름

사유: 법령 준수 문구가 포함되어 있어 법적 기준에 부합하며 안전합니다.

근거: 근로기준법 제15조""",
        "legal_guide": """1. 근로시간 항목

참조 법령: 「근로기준법」 제50조

요지: 1일 근로시간은 8시간, 1주 근로시간은 40시간을 초과할 수 없습니다.

2. 근무일·휴일 항목

참조 법령: 「근로기준법」 제55조

요지: 사용자는 근로자에게 1주에 평균 1회 이상의 유급휴일을 보장하여야 합니다.

3. 임금 항목

참조 법령: 「최저임금법」 제6조, 「근로기준법」 제43조

요지: 사용자는 근로자에게 고용노동부 장관이 고시하는 최저임금 이상을 지급해야 하며(최저임금법 제6조), 임금은 통화로, 직접, 전액을, 매월 1회 이상 일정한 날짜에 지급해야 합니다(근로기준법 제43조)."""
    }
}


# --- 3. EXAMPLE SCENARIOS ---
EXAMPLES = {
    "선택하세요": None,
    "예시 1: 임산부 항만하역업": {
        "description": "30세, 여성, 임신 중 - 유해업무 금지 위반",
        "profile": {
            "A1": "내국인", "A2": "여성", "A2_2": "임산부 또는 출산 후 1년 이내", 
            "A3": "만 18세 이상 ~ 만 60세 미만", "A4": "비장애인", "A5": "일반(해당없음)"
        },
        "text": """1. 근로개시일: 2025년 1월 18일
2. 근무장소: 갑 소유의 A 선박
3. 업무의 내용: 항만하역업
4. 소정근로시간: 10:00~13:30 (휴게 30분 포함, 주 12시간)
5. 근무일·휴일: 미기재
6. 임금: 시급 11,000원, 매월 20일 지급, 계좌이체
7. 연차유급휴가: 근로기준법에 따라 부여
8. 사회보험 적용여부: 산재보험, 건강보험 적용
9. 근로계약서 교부: 계약 체결 시 교부
10. 근로계약·취업규칙 성실 이행: 성실 이행 의무 명시
11. 그 밖의 사항: 근로관계법령에 따름""",
        "model_answer_key": "example_pregnant"
    },
    "예시 2: 부사장 포괄임금제": {
        "description": "35세, 남성 - 임원 명칭 사용 및 포괄임금제",
        "profile": {
            "A1": "내국인", "A2": "남성", "A2_2": "해당 없음", 
            "A3": "만 18세 이상 ~ 만 60세 미만", "A4": "비장애인", "A5": "포괄임금제"
        },
        "text": """1. 근로개시일: 2025년 1월 1일부터 (수습기간 3개월, 1년 단위 자동연장)
2. 근무장소: 본사에서 지정한 장소. 단, 담당업무 및 근무 장소는 회사 사정에 따라 변경할 수 있다.
3. 업무의 내용: 부사장
4. 소정근로시간: 09시 00분 ~ 18시 00분 (휴게: 12시 00분 ~ 13시 00분) (1일 8시간, 1주 40시간)
5. 근무일·휴일: 매주 5일 근무, 주휴일 매주 일요일, 공휴일은 근로기준법이 정하는 바에 따름
6. 임금: 연봉 50,000,000원, 수습기간 중 100% 지급. 상여금 없음. 그 밖의 수당 없음. 매월 5일 임금 지급, 계좌 입금
7. 연차유급휴가: 근로기준법에서 정하는 바에 따라 부여함
8. 사회보험 적용여부: 4대 사회보험 적용(가입)을 원칙으로 함
9. 근로계약서 교부: 계약 체결 시 교부
10. 근로계약·취업규칙 성실 이행: 성실 이행 의무 명시, 자리를 비울 때에는 반드시 상사에게 보고하고 컨펌 하에 움직이도록 한다.
11. 그 밖의 사항: 근로관계법령에 따름""",
        "model_answer_key": "example_executive"
    },
    "예시 3: 카페 아르바이트": {
        "description": "19세, 남성 - 휴게시간 미보장 및 주휴일 누락",
        "profile": {
            "A1": "내국인", "A2": "남성", "A2_2": "해당 없음", 
            "A3": "만 18세 이상 ~ 만 60세 미만", "A4": "비장애인", "A5": "포괄임금제"
        },
        "text": """1. 근로개시일: 2025년 1월 1일부터 12월 31일까지 (단, 3개월은 수습기간으로 한다.)
2. 근무장소: 서울특별시 종로구 A카페
3. 업무의 내용: 카페 업무 전반
4. 소정근로시간: 08:30~12:30 (휴게시간은 조기퇴근으로 대신한다.)
5. 근무일·휴일: 주 4일(월, 화, 목, 금) 근무, 주휴일 없음
6. 임금: 시급 11,000원, 상여금 없음, 그 밖의 수당 없음, 매월 27일 지급, 계좌 입금, 수습기간의 임금은 시급의 80%로 정한다.
7. 연차유급휴가: 단시간근무자는 연차유급휴가를 지급하지 않는다.
8. 사회보험 적용여부: 4대 사회보험 적용(가입)을 원칙으로 함
9. 근로계약서 교부: 계약 체결 시 교부
10. 근로계약·취업규칙 성실 이행: 사업주와 근로자는 각자가 근로계약, 취업규칙, 단체협약을 지키고 성실하게 이행하여야 함
11. 그 밖의 사항: 없음""",
        "model_answer_key": "example_cafe"
    },
    "예시 4: 일반근로": {
        "description": "53세, 여성, 일반근로 - 최저임금 미달 및 근로시간 초과",
        "profile": {
            "A1": "내국인", "A2": "여성", "A2_2": "해당 없음", 
            "A3": "만 18세 이상 ~ 만 60세 미만", "A4": "비장애인", "A5": "일반(해당없음)"
        },
        "text": """1. 근로개시일: 2025년 1월 1일
2. 근무장소: 회사 필요에 따라 전국 사업장 어디든 발령 가능
3. 업무의 내용: 회사 지시하는 모든 업무 일체
4. 소정근로시간: 08:00~18:00(휴게 1시간), 1일 9시간·주 45시간
5. 근무일·휴일: 주 5일 근무/주휴일: 일요일(무급)
6. 임금: 월 2,000,000원
7. 연차유급휴가: 근로기준법에 따라 부여
8. 사회보험 적용여부: 4대 사회보험(고용보험, 산재보험, 국민연금, 건강보험) 적용(가입)을 원칙으로 함
9. 근로계약서 교부: 계약 체결 시 교부
10. 근로계약·취업규칙 성실 이행: 성실 이행 의무 명시
11. 그 밖의 사항: 근로관계법령에 따름""",
        "model_answer_key": "example_53yo"
    }
}

# --- 4. LOADING FUNCTIONS ---
def load_labeling_manual():
    if os.path.exists(CSV_FILE):
        try:
            df = pd.read_csv(CSV_FILE, encoding='utf-8')
        except UnicodeDecodeError:
            try: df = pd.read_csv(CSV_FILE, encoding='cp949')
            except: return "CSV Error"
        
        txt = ""
        for _, row in df.iterrows():
            txt += f"- [조항 {row.get('조항번호', '?')}] 조건: {row.get('조건(소전제)', 'N/A')} -> 판정: {row.get('라벨링(결론)', '미정')}, 사유: {row.get('사유(대전제)', '')}\n"
        return txt
    return "매뉴얼 없음"

@st.cache_resource(show_spinner=True)
def init_rag_system():
    """근로기준법과 취업규칙 PDF를 모두 로드하여 통합 RAG 시스템 초기화"""
    pdf_files = {
        "근로기준법": "근로기준법.pdf",
        "취업규칙": "취업규칙.pdf"
    }
    
    all_splits = []
    loaded_files = []
    
    for name, pdf_path in pdf_files.items():
        if not os.path.exists(pdf_path):
            st.warning(f"⚠️ {pdf_path} 파일을 찾을 수 없습니다.")
            continue
            
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            
            for doc in docs:
                doc.metadata['source_name'] = name
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n제", "\n\n", "\n", " ", ""]
            )
            splits = text_splitter.split_documents(docs)
            all_splits.extend(splits)
            loaded_files.append(name)
            
        except Exception as e:
            st.warning(f"⚠️ {name} 로딩 오류: {e}")
    
    if not all_splits:
        st.error("❌ RAG 시스템 초기화 실패")
        return None
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        vectorstore = FAISS.from_documents(all_splits, embeddings)
        st.success(f"✅ RAG 시스템 로드 완료: {', '.join(loaded_files)}")
        return vectorstore
        
    except Exception as e:
        st.error(f"❌ 벡터스토어 생성 오류: {e}")
        return None

def parse_contract_to_chunks(text):
    """
    계약서 텍스트를 조항별로 분리하고, 깨진 시간 포맷(예: 12시00분20시00분)을 자동 보정합니다.
    """
    import re
    
    # --- [추가된 부분] 시간/숫자 포맷 클리닝 ---
    # 1. 붙어있는 시간 분리 (예: "12시 00분20시 00분" -> "12시 00분 ~ 20시 00분")
    # 원리: '00분' 뒤에 바로 숫자+'시'가 오면 중간에 ' ~ '를 삽입
    text = re.sub(r'(\d+분)\s*(\d+시)', r'\1 ~ \2', text)
    
    # 2. 휴게시간 등에서 분 단위 없이 붙은 경우 (예: "12:0013:00")
    text = re.sub(r'(\d+:\d+)\s*(\d+:\d+)', r'\1 ~ \2', text)

    # 3. 괄호 사이 공백 확보 (가독성 향상)
    text = text.replace(")(", ") (")
    # ------------------------------------------

    # 1. 줄바꿈 정규화
    text = text.replace("\r\n", "\n")
    
    # 2. 조항 분리 패턴 (숫자+점, 제N조, 괄호숫자 등)
    pattern = r'(?m)^(\s*(?:제)?\s*\d+(?:\.|조|\)|\])\s*|^\s*\(\d+\)\s*)'
    
    parts = re.split(pattern, text)
    
    clauses = []
    
    # 서두 처리
    if parts[0].strip():
        clauses.append({"id": "0", "label": "서두", "content": parts[0].strip()})
    
    # 조항 정리
    for i in range(1, len(parts), 2):
        label_marker = parts[i].strip()
        content = parts[i+1].strip()
        
        # 라벨에서 숫자만 추출 (ID용)
        clean_id = re.sub(r'[^\d]', '', label_marker)
        if not clean_id: clean_id = str(i)
        
        clauses.append({
            "id": clean_id,
            "label": label_marker,
            "content": content if content else "(내용 없음)"
        })
        
    return clauses

# --- 5. AI ENGINE WITH STRICT FORMAT ---
def run_ai_analysis_body(profile, contract_text, labeling_manual, model_name, api_key, vectorstore=None, progress_callback=None):
    """
    [수정됨] 강력한 한국어 출력 강제 및 포맷 준수 로직 적용
    """
    if progress_callback: progress_callback(PROGRESS_STEPS[0], 10) # 1. Profile Analysis

    os.environ["TOGETHER_API_KEY"] = api_key

    try:
        if progress_callback: progress_callback(PROGRESS_STEPS[1], 20) # 2. Parsing
        
        # 1. 파싱 (이전 단계에서 추가한 함수 사용)
        parsed_clauses = parse_contract_to_chunks(contract_text)
        
        # 프롬프트 입력용 텍스트 구성
        formatted_contract_input = ""
        for clause in parsed_clauses:
            formatted_contract_input += f"[[조항번호 {clause['label']}]]\n내용: {clause['content']}\n\n"

        if not parsed_clauses:
            formatted_contract_input = f"[[전체 텍스트]]\n{contract_text}"

        # 2. 모델 설정 (temperature를 0으로 낮춰서 창의성(영어발산) 억제)
        llm = ChatTogether(
            model=model_name,
            together_api_key=api_key,
            temperature=0.1, 
            max_tokens=1500, 
            stop=["<|eot_id|>", "<|end_of_text|>"]
        )

        if progress_callback: progress_callback(PROGRESS_STEPS[2], 30) # 3. DB Search

        # 3. RAG 검색 (토큰 절약을 위해 핵심만)
        rag_context = ""
        if vectorstore:
            queries = ["근로계약서 필수 기재 사항", "근로기준법 위반"]
            if profile.get("A2_2") != "해당 없음": queries.append("임산부 근로 보호")
            
            for i, query in enumerate(queries):
                if progress_callback:
                    progress = 30 + int((i / len(queries)) * 20)
                    progress_callback(f"{PROGRESS_STEPS[2]} - {query} 검색 중...", progress)
                
                docs = vectorstore.similarity_search(query, k=2)
                for doc in docs:
                    # 줄바꿈 제거하여 토큰 절약
                    content_clean = doc.page_content.replace("\n", " ")[:200]
                    rag_context += f"- {content_clean}\n"
        
        if progress_callback: progress_callback(PROGRESS_STEPS[4], 75) # 5. Legal Matching

        # 4. [중요] 시스템 프롬프트: 한국어 강제 제약조건 추가
        system_prompt = f"""당신은 대한민국 노동법 전문 AI입니다. 아래 지침을 엄격히 따르십시오.

[매우 중요]
1. **모든 답변은 반드시 '한국어(Korean)'로만 작성하십시오.**
2. 'Input', 'Reason', 'Risk', 'Reference' 등의 영어 단어를 절대 사용하지 마십시오.
3. 반드시 아래 [필수 출력 형식]의 한글 키워드(위험도, 입력, 사유, 근거, 개선)를 그대로 사용하십시오.
4. 반드시 아래 [필수 출력 형식]의 근거를 설명하십시오.

[분석 기준]
- 사용자 프로필: {profile}
- 법령 정보: {rag_context}
- 판정 매뉴얼: {labeling_manual}

[전체 요약]
### 계약서 전체 분석 요약
(전체적인 총평을 한국어로 3문장 이내 요약)

[조항별 세부 분석]
(아래의 [필수 출력 형식]에 맞춰 각 조항별로 번호와 위험도를 매겨 분석 결과 나열)

[필수 출력 형식]
반드시 위 분석 기준을 바탕으로, 각 조항에 대해 아래 형식을 그대로 사용하여 한국어로 답변하십시오:

[조항 제목] ([위험도: 저위험/중위험/고위험])
- 📥 **입력**: (계약서 내용을 수정 없이 인용)
- 🔍 **사유**: (위험도 판정 이유를 한국어로 상세히 설명)
- ⚖️ **근거**: (관련 법령명. 예: 근로기준법 제17조)
- ✅ **개선**: (저위험이면 "없음" 출력, 그 외에는 한국어로 개선사항 제안)

"""

        # LangChain 변수 충돌 방지를 위한 이스케이프 처리
        system_prompt_escaped = system_prompt.replace("{", "{{").replace("}", "}}")

        # 프롬프트 체인 생성
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt_escaped),
            ("human", """[분석할 계약서 내용]
{contract_input}

위 내용을 한국어로 분석해 주세요.""")
        ])

        chain = prompt | llm

        if progress_callback: progress_callback(PROGRESS_STEPS[5], 85) # 6. Improvement
        
        # 실행
        response = chain.invoke({"contract_input": formatted_contract_input})

        if progress_callback: progress_callback(PROGRESS_STEPS[6], 100) # 7. Complete

        return response.content

    except Exception as e:
        return f"Error: {str(e)}"

def get_fixed_advice_text(profile):
    """
    Returns the legal advice text based on the user's profile conditions 
    defined in the 'gojeong-sancul-dabbyeon.PDF'.
    """
    advice_list = []

    # A1-1: 내국인
    if profile.get("A1") == "내국인":
        advice_list.append("""
1. 내국인 근로자 적용 법령
- 적용 법령: 근로기준법 제17조, 제43조
- 요지: 
  - 사용자는 근로계약 체결 시 근로조건(임금·근로시간 등)을 서면 명시해야 함 (제17조).
  - 임금은 통화로, 직접, 전액을, 매월 1회 이상 일정한 날짜에 지급해야 함 (제43조).
- 요약: 내국인은 근로기준법 및 최저임금법의 일반적 근로조건이 적용됩니다.
""")

    # A1-2: 외국인
    elif profile.get("A1") == "외국인":
        advice_list.append("""
1. 외국인 근로자 적용 법령
- 적용 법령: 외국인근로자의 고용 등에 관한 법률 제6조~제12조, 출입국관리법 제18조
- 요지: 
  - 고용허가제에 따라 고용주가 허가를 받아야 하며, 체류자격(E-9 등)에 따라 근로계약 가능 여부가 결정됨.
  - 체류자격 외 취업은 불법.
- 요약: 외국인 근로자는 고용허가제 절차 및 체류자격 확인이 필수입니다.
""")

    # A2-1: 여성 (General)
    if profile.get("A2") == "여성":
        advice_list.append("""
2. 여성 근로자 보호
- 적용 법령: 근로기준법 제65조~제74조
- 요지: 
  - 제65조: 임신·출산 여성의 유해·위험 업무 금지.
  - 제74조: 출산 전후 휴가(90일, 다태아 120일) 및 유산·사산휴가 부여.
- 요약: 여성 근로자는 유해·위험 업무 제한, 출산휴가 및 보호를 받을 권리가 있습니다.
""")

    # A2-2: 남성
    if profile.get("A2") == "남성":
        advice_list.append("""
2. 남성 근로자 적용 기준
- 적용 법령: 근로기준법 제50조, 제55조
- 요지: 
  - 제50조: 1일 8시간, 주 40시간 초과 불가.
  - 제55조: 1주 1회 이상 유급휴일 보장.
- 요약: 남성 근로자는 일반 근로시간 및 휴일 규정이 적용됩니다.
""")

    # A2-2-1: 임산부
    if profile.get("A2_2") == "임산부 또는 출산 후 1년 이내":
        advice_list.append("""
2-1. 임산부 보호 특별 규정
- 적용 법령: 근로기준법 제65조, 제70조~제74조, 남녀고용평등법 제19조
- 요지: 
  - 제65조: 임산부는 유해·위험 사업 사용 금지.
  - 제71조: 임신 중 시간외근로 금지.
  - 제74조: 출산휴가급여, 출산전후휴가 90일(다태아 120일) 부여.
  - 남녀고용평등법 제19조: 육아휴직 보장.
- 요약: 임산부는 유해·위험 업무 금지, 시간외근로 금지, 출산휴가 및 출산전후휴가급여 보장 대상입니다.
""")
        
    # A2-2-2: 비임산부 여성
    elif profile.get("A2") == "여성" and profile.get("A2_2") != "임산부 또는 출산 후 1년 이내":
         advice_list.append("""
2-1. 여성 유해사업 사용 금지
- 적용 법령: 근로기준법 제65조
- 요지: 사용자는 임산부가 아닌 18세 이상 여성이라도 임신·출산 기능에 유해한 사업에 사용 불가.
- 요약: 일반 여성 근로자도 보건상 유해한 사업에 사용할 수 없습니다.
""")

    # A3-1: 연소자 (만 18세 미만)
    if profile.get("A3") == "만 18세 미만":
        advice_list.append("""
3. 연소 근로자 보호
- 적용 법령: 근로기준법 제64조~제70조, 청소년보호법 제29조
- 요지: 
  - 제64조: 만15세 미만 고용 금지(취직인허증 예외).
  - 제69조: 1일 7시간, 주 35시간 제한.
  - 제70조: 야간·휴일근로 금지.
- 요약: 연소근로자는 근로시간·업종 제한, 취직인허증 필요, 야간·휴일근로 금지됩니다.
""")

    # A3-2: 일반 성인 (만 18~60)
    elif profile.get("A3") == "만 18세 이상 ~ 만 60세 미만":
        advice_list.append("""
3. 일반 근로자 (연령)
- 적용 법령: 근로기준법 제50조, 근로기준법 제53조
- 요지: 1일 8시간, 주 40시간 초과 금지. 연장근로는 주 12시간 한도.
- 요약: 일반 근로자 기준이 적용됩니다.
""")

    # A3-3: 고령자 (만 60세 이상)
    elif profile.get("A3") == "만 60세 이상":
        advice_list.append("""
3. 고령 근로자
- 적용 법령: 고령자고용촉진법 제19조, 제21조
- 요지: 
  - 제19조: 정년 후 재고용 노력 의무.
  - 제21조: 임금피크제 등 고령 근로자 근로조건 완화 가능.
- 요약: 고령 근로자는 정년 후 재고용 및 임금조정 규정의 적용 대상입니다.
""")

    # A4-1: 장애인
    if profile.get("A4") == "장애인":
        advice_list.append("""
4. 장애인 근로자 보호
- 적용 법령: 장애인고용촉진법 제5조, 장애인차별금지법 제10~12조
- 요지: 
  - 제5조: 장애인의 능력을 정당하게 평가하고 적정 고용 관리 의무.
  - 제11조: 정당한 편의 제공(시설·장비, 근무시간 조정 등).
- 요약: 장애인 근로자는 차별을 받지 않으며 정당한 편의를 제공받을 권리가 있습니다.
""")

    # [NEW] A4-2: 비장애인
    elif profile.get("A4") == "비장애인":
        advice_list.append("""
4. 비장애인 근로자
- 적용 법령: 근로기준법 제17조
- 요지: 일반 근로조건 명시 조항 적용.
- 요약: 비장애인 근로자는 일반 기준에 따라 보호됩니다.
""")

    # A5-1: 포괄임금제
    if profile.get("A5") == "포괄임금제":
        advice_list.append("""
5. 포괄임금제 유의사항
- 적용 법령: 근로기준법 제56조
- 요지: 연장·야간·휴일근로는 통상임금의 50% 이상 가산 지급해야 함.
- 요약: 포괄임금제라도 수당 포함 여부를 명시해야 합니다.
""")

    # A5-2: 유연근무제
    elif profile.get("A5") == "유연근무제":
        advice_list.append("""
5. 유연근무제
- 적용 법령: 근로기준법 제52조
- 요지: 근로자대표와 서면합의 필요. 1개월 단위 평균 주40시간 초과 금지.
- 요약: 유연근무제는 서면합의가 필수이며 정산기간 기준을 지켜야 합니다.
""")

    # [NEW] A5-3: 교대근무제
    elif profile.get("A5") == "교대근무제":
        advice_list.append("""
5. 교대근무제
- 적용 법령: 근로기준법 제59조 제2항
- 요지: 근로일 종료 후 다음 근로일까지 11시간 이상 연속휴식 보장해야 함.
- 요약: 교대근무자는 11시간 연속휴식이 의무입니다.
""")

    # [NEW] A5-4: 특별한 근로시간 유형 없음 (일반)
    else: # Default or explicitly "일반(해당없음)"
        advice_list.append("""
5. 일반 근로시간 유형
- 적용 법령: 근로기준법 제50조
- 요지: 1일 8시간, 1주 40시간 기준. 연장 시 근로자 동의 필요.
- 요약: 일반 법정근로시간 기준을 적용합니다.
""")

    return "\n".join(advice_list) if advice_list else "해당하는 특별 법령 가이드가 없습니다."


# --- 6. PROGRESS BARS ---

def show_fake_progress():
    """모범 답변용 가짜 프로그레스 바"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_steps = len(PROGRESS_STEPS)
    
    for i, step_name in enumerate(PROGRESS_STEPS):
        status_text.text(f"⚙️ {step_name}")
        # Progress from 0 to 100 based on steps
        progress_val = int(((i + 1) / total_steps) * 100)
        progress_bar.progress(progress_val)
        time.sleep(1.0)  # Demo speed
    
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()


def show_real_progress(callback_steps):
    """실제 AI 분석용 프로그레스 바 - callback 기반"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_progress(step_name, progress_value):
        """Called by AI analysis function to update progress"""
        status_text.text(f"⚙️ {step_name}")
        progress_bar.progress(progress_value)
    
    try:
        result = callback_steps(update_progress)
        progress_bar.progress(100)
        status_text.text("✅ 분석 완료!")
        time.sleep(1)
        return result
    finally:
        progress_bar.empty()
        status_text.empty()


# --- 7. UI LAYOUT ---
with st.sidebar:
    st.header("⚙️ 설정")
    api_key = st.text_input("Together API Key", value=DEFAULT_API_KEY, type="password")
    st.divider()
    model_name = st.text_input("Model ID", value=DEFAULT_MODEL)
    st.caption("Fine-Tuned Llama-3")
    st.divider()
    if st.button("🔄 처음부터 다시 하기"):
        st.session_state.step = "A"
        st.session_state.user_profile = {}
        st.session_state.contract_text = ""
        st.session_state.selected_example = None
        st.session_state.original_contract_text = ""
        st.rerun()

# MAIN LOGIC
if not api_key: st.stop()

# Load Manual only
manual_text = load_labeling_manual()

# STEP A: PROFILE
if st.session_state.step == "A":
    st.title("Step A. 근로자 정보 입력")
    
    p = st.session_state.user_profile if st.session_state.user_profile else {
        "A1":"내국인",
        "A2":"남성",
        "A2_2":"해당 없음",
        "A3":"만 18세 이상 ~ 만 60세 미만",
        "A4":"비장애인",
        "A5":"일반(해당없음)"
    }
    
    c1, c2 = st.columns(2)
    
    with c1:
        a1 = st.radio("A1. 국적", ["내국인", "외국인"], index=0 if p["A1"]=="내국인" else 1)
        a2 = st.radio("A2. 성별", ["여성", "남성"], index=0 if p["A2"]=="여성" else 1)
        a2_2 = "해당 없음"
        if a2 == "여성":
            a2_2 = st.radio("↳ 상세", ["임산부 또는 출산 후 1년 이내", "해당 없음"], index=0 if p["A2_2"]!="해당 없음" else 1)
    
    with c2:
        a3 = st.radio("A3. 연령", ["만 18세 미만", "만 18세 이상 ~ 만 60세 미만", "만 60세 이상"], index=1)
        a4 = st.radio("A4. 장애", ["장애인", "비장애인"], index=1)
        a5 = st.radio("A5. 근무 유형", ["포괄임금제", "유연근무제", "교대근무제", "일반(해당없음)"], index=3)
    
    st.divider()
    
    if st.button("다음 >", type="primary"):
        st.session_state.user_profile = {"A1":a1, "A2":a2, "A2_2":a2_2, "A3":a3, "A4":a4, "A5":a5}
        # [NEW] Save user profile for backup so we can restore it if they cancel an example
        st.session_state.saved_user_profile = st.session_state.user_profile.copy()
        st.session_state.step = "B"
        st.rerun()


# STEP B: CONTRACT
elif st.session_state.step == "B":
    st.title("Step B. 계약서 조항 입력")
    
    # [NEW] Display selected options from Step A
    p = st.session_state.user_profile
    st.info(f"입력하신 내용은 A1: {p.get('A1', '-')} A2: {p.get('A2', '-')} A2-2: {p.get('A2_2', '-')} A3: {p.get('A3', '-')} A4: {p.get('A4', '-')} A5: {p.get('A5', '-')} 입니다.")
    
    c_in, c_guide = st.columns([2, 1])
    
    with c_guide:
        st.subheader("📋 예시 계약서 선택")
        # Ensure session state for selection exists to handle reset logic
        if "prev_sel" not in st.session_state:
            st.session_state.prev_sel = "선택하세요"

        sel = st.selectbox("예시 선택", list(EXAMPLES.keys()))
        
        # [NEW] Logic to reset if user goes back to "선택하세요"
        if sel == "선택하세요" and st.session_state.prev_sel != "선택하세요":
            st.session_state.contract_text = ""
            st.session_state.selected_example = None
            st.session_state.original_contract_text = ""
            
            # [NEW] Restore user profile from Step A
            if "saved_user_profile" in st.session_state:
                st.session_state.user_profile = st.session_state.saved_user_profile.copy()
            
            st.session_state.prev_sel = "선택하세요"
            st.rerun()

        st.session_state.prev_sel = sel

        if sel != "선택하세요":
            case = EXAMPLES[sel]
            st.info(case['description'])
            
            if st.button("✅ 이 예시 적용", type="primary"):
                st.session_state.contract_text = case['text']
                st.session_state.user_profile = case['profile']
                st.session_state.selected_example = case.get('model_answer_key')
                st.session_state.original_contract_text = case['text'] 
                st.success("예시가 적용되었습니다!")
                time.sleep(1)
                st.rerun()
    
    with c_in:
        # Check if an example is active to disable editing
        is_example_active = (st.session_state.selected_example is not None)
        
        # [NEW] Added placeholder text & disabled state
        txt = st.text_area(
            "계약서 내용", 
            value=st.session_state.contract_text, 
            height=400, 
            help="정확한 분석을 위해 조항 번호를 포함하여 입력해주세요.",
            # UPDATE THIS PLACEHOLDER
            placeholder="[정확한 분석을 위한 입력 가이드]\n각 조항을 줄바꿈하여 구분해 주세요.\n\n1. 근로계약기간: 2024.01.01 ~ \n2. 근무장소: 서울시...\n3. 업무내용: ...\n\n(위와 같이 번호를 붙여주시면 AI가 더 정확하게 인식합니다)",
            disabled=is_example_active 
        )
        
        if st.button("분석 시작", type="primary"):
            if not txt.strip():
                st.error("계약서 내용을 입력해주세요.")
            else:
                st.session_state.contract_text = txt
                
                # Check if user modified the text (Only relevant if disabled=False, but kept for safety)
                if st.session_state.get('original_contract_text') and \
                   txt != st.session_state.original_contract_text:
                    st.session_state.selected_example = None 
                    st.info("계약서 수정이 감지되었습니다.")
                    time.sleep(1)
                
                st.session_state.step = "C"
                st.rerun()


# STEP C: ANALYSIS OUTPUT
elif st.session_state.step == "C":
    st.title("[근로계약서 자동 분석 결과]")
    
    # 1. FIXED DISCLAIMER
    st.markdown("### 개발 중인 내용 및 변호사법 준수 안내")
    st.info("""
본 인공지능은 근로계약서의 주요 조항을 자동으로 분석하여, 관련 법령 및 판례를 반영한 기준에 따른 조항별 리스크를 진단하고 개선방향을 제시하는 시스템입니다.

현재 시스템에는 근로기준법 등 일부 법률과 주요 판례만 반영된 상태이며, 사용자(사업주·근로자) 입장별 맞춤 분석 기능은 개발 중입니다. 현재 단계에서는 근로자 입장을 기준으로 위험도와 개선 방향을 산출하고 있습니다.

본 시스템은 「변호사법」 제34조 제5항 및 제109조를 준수하며, 비변호사가 법률사무를 수행하거나 이를 알선하지 않도록 설계되어 있습니다. 본 결과는 법률 자문이 아닌 참고용 분석 자료임을 고려해주시기 부탁드립니다.

계약으로 인해 실제 분쟁이 발생하여 법적 해석이 필요한 경우에는 반드시 변호사 등 법률 전문가의 자문을 받으시기 바랍니다.
""")
    
    p = st.session_state.user_profile
    model_answer_key = st.session_state.get('selected_example')
    
    # Check if exact match with example
    is_original_contract = (
        model_answer_key and 
        model_answer_key in MODEL_ANSWERS and
        st.session_state.contract_text == st.session_state.get('original_contract_text', '')
    )
    
    # [EDITED] Always Init RAG (removed the 'if not is_original_contract' check)
    # This ensures the loading spinner and success message appear even for model answers.
    with st.spinner("📚 근로기준법 및 취업규칙 데이터베이스 로딩 중..."):
        vectorstore = init_rag_system()

    def format_details(text):
        # Format headers: "1. Title (Risk)" -> "1. Title ([위험도: Risk])"
        text = re.sub(r'(\d+\..*?)\s*\((저위험|중위험|고위험)\)', r'\1 ([위험도: \2])', text)
        # Add emojis and bolding for standard fields
        text = re.sub(r'\n\s*입력:', r'\n- 📥 **입력**:', text)
        text = re.sub(r'\n\s*사유:', r'\n- 🔍 **사유**:', text)
        text = re.sub(r'\n\s*근거:', r'\n- ⚖️ **근거**:', text)
        text = re.sub(r'\n\s*개선:', r'\n- ✅ **개선**:', text)
        return text
    
    if is_original_contract:
        # === MODEL ANSWER: USE FAKE PROGRESS ===
        show_fake_progress()
        
        # USE MODEL ANSWER
        model_answer = MODEL_ANSWERS[model_answer_key]
    
        st.write(f"현재 입력된 계약서에서 이용자는 **\"{model_answer['profile_display']}\"** 으로 확인됩니다.")
        st.divider()

        # 2. Construct a single body string that mimics the AI output structure
        formatted_details = format_details(model_answer['details'])
        ai_body = f"### 계약서 전체 분석 요약\n{model_answer['summary']}\n\n### 조항별 세부 분석\n{formatted_details}"

        # 3. Calculate actual counts using Regex (Same as AI logic)
        high_risk = len(re.findall(r'위험도:\s*.*고위험', ai_body))
        med_risk = len(re.findall(r'위험도:\s*.*중위험', ai_body))
        low_risk = len(re.findall(r'위험도:\s*.*저위험', ai_body))

        # 4. Display accurate counts at the top
        c1, c2, c3 = st.columns(3)
        c1.metric("🔴 고위험", f"{high_risk}건")
        c2.metric("🟠 중위험", f"{med_risk}건")
        c3.metric("🟢 저위험", f"{low_risk}건")

        # 5. Display the text content with Colors
        colored_body = ai_body \
            .replace("고위험", ":red[**고위험**]") \
            .replace("중위험", ":orange[**중위험**]") \
            .replace("저위험", ":green[**저위험**]")
    
        st.markdown(colored_body)
        # --- FORMATTING LOGIC END ---

        st.divider()

        st.markdown("### 법령 참조 안내")
        st.write(f"현재 입력된 계약서에 따라 다음 항목이 자동으로 제안됩니다. 이는 귀하의 계약 상황({model_answer['profile_display']})에 직접적으로 관련된 법령 조항이므로 반드시 확인하십시오.")
    
        with st.container(border=True):
            st.text(model_answer['legal_guide'])

    
    else:
        # === AI GENERATION: REAL-TIME SPINNER ===
        profile_desc = f"\"{p['A1']}, {p['A2']}"
        if p['A2'] == "여성" and p['A2_2'] != "해당 없음":
            profile_desc += f"({p['A2_2']})"
        profile_desc += f", {p['A3']}, {p['A4']}, {p['A5']}\""
        
        st.write(f"현재 입력된 계약서에서 이용자는 **{profile_desc}** 으로 확인됩니다.")
        st.divider()
        
        # Define wrapper for callback
        def run_with_progress(progress_callback):
            return run_ai_analysis_body(
                p, 
                st.session_state.contract_text, 
                manual_text, 
                model_name, 
                api_key, 
                vectorstore=vectorstore, 
                progress_callback=progress_callback
            )

        try:
            # CALL REAL PROGRESS
            ai_body = show_real_progress(run_with_progress)
            
            # --- FIX 3: CHECK FOR EMPTY OUTPUT ---
            if not ai_body or ai_body.strip() == "":
                 st.error("⚠️ AI 분석 결과가 비어있습니다. 다시 시도해주세요.")
                 st.caption("가능한 원인: 모델이 응답을 생성하다가 중단되었거나, 입력 내용이 너무 깁니다.")

            else:
                # 1. Calculate actual counts using Regex
                import re
                ai_body = format_details(ai_body)
                high_risk = len(re.findall(r'위험도:\s*.*고위험', ai_body))
                med_risk = len(re.findall(r'위험도:\s*.*중위험', ai_body))
                low_risk = len(re.findall(r'위험도:\s*.*저위험', ai_body))
    
                # 2. Display accurate counts at the top
                c1, c2, c3 = st.columns(3)
                c1.metric("🔴 고위험", f"{high_risk}건")
                c2.metric("🟠 중위험", f"{med_risk}건")
                c3.metric("🟢 저위험", f"{low_risk}건")

                # 3. Display the text content
                colored_body = ai_body \
                    .replace("고위험", ":red[**고위험**]") \
                    .replace("중위험", ":orange[**중위험**]") \
                    .replace("저위험", ":green[**저위험**]")
                
                st.markdown(colored_body)
            
        except Exception as e:
            st.error(f"분석 중 오류가 발생했습니다: {e}")

        
        st.divider()
        
        st.markdown("### 법령 참조 안내")
        st.write(f"현재 입력된 계약서에 따라 다음 항목이 자동으로 제안됩니다. 이는 귀하의 계약 상황({profile_desc})에 직접적으로 관련된 법령 조항이므로 반드시 확인하십시오.")
        
        advice_content = get_fixed_advice_text(p)
        with st.container(border=True):
            st.text(advice_content)
    
    st.divider()
    
    # 6. FIXED FOOTER
    st.markdown("### 법적 고지")
    st.warning("""
본 분석은 「변호사법」 제34조 제5항 및 제109조에 따라 법률사무 또는 자문행위에 해당하지 않으며,
법령상 기준에 따른 계약서 리스크 진단 및 정보 제공의 목적만을 가집니다.

법적 분쟁, 행정 신고, 분쟁 대응 등의 상황에서는 반드시 변호사 또는 고용노동부의 공식 자문을 받으시기 바랍니다.
""")
