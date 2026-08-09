"""
피터 린치 AI 한국주식 시뮬레이터 (실데이터 연동판)

데이터 소스
- 가격        : FinanceDataReader (KRX 실제 종가)
- PER         : pykrx (일자별 실제 PER, KRX 산출 기준)
- 성장률/부채비율 : OpenDartReader (DART 공시 재무제표, 실제 매출/영업이익 YoY, 부채비율)
- 뉴스        : 네이버 금융 종목뉴스 페이지 크롤링 (실제 기사 제목 + 날짜)

주의 (반드시 읽어주세요)
1. 이 코드는 이 환경(샌드박스)의 네트워크 제한 때문에 실제 통신 테스트를 하지 못했습니다.
   finance.naver.com, opendart.fss.or.kr 등 외부 사이트 접근이 이 환경에서는 막혀 있어서
   문법(syntax) 검증만 했고, 실제 실행은 본인 컴퓨터(또는 자유로운 네트워크 환경)에서
   꼭 해보고 결과가 이상하면 알려주세요. 특히 네이버 페이지 구조는 언제든 바뀔 수 있어서
   크롤링 부분이 깨지면 셀렉터(CSS selector)를 다시 맞춰야 할 수도 있습니다.
2. DART 재무제표 조회에는 무료 API 키가 필요합니다. https://opendart.fss.or.kr 에서
   회원가입 후 발급받으세요 (즉시 발급, 무료).
3. 분기 재무제표는 "공시된 시점"을 기준으로만 사용합니다 (예: 1분기 실적은 보통 5월 중순
   이후에나 공개되므로, 4월에는 아직 그 데이터를 못 봤다고 가정) - look-ahead bias 방지.
4. 네이버 뉴스 크롤링은 서버에 부담을 주지 않도록 요청 사이에 딜레이를 넣었습니다.
   과도하게 자주 돌리지 마세요.

실행 전 설치:
    pip install streamlit pandas numpy plotly finance-datareader pykrx OpenDartReader \
        requests beautifulsoup4 openai --break-system-packages

실행:
    streamlit run peter_lynch_ai_simulator_v2.py
"""

import time
import json
import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ---- 선택적 의존성 처리 (없으면 화면에 경고만 띄우고 폴백 동작) ----
try:
    import FinanceDataReader as fdr
    FDR_AVAILABLE = True
except ImportError:
    FDR_AVAILABLE = False

try:
    from pykrx import stock as pykrx_stock
    PYKRX_AVAILABLE = True
except ImportError:
    PYKRX_AVAILABLE = False

try:
    import OpenDartReader
    DART_LIB_AVAILABLE = True
    DART_LIB_IMPORT_ERROR = None
except ImportError as e:
    DART_LIB_AVAILABLE = False
    DART_LIB_IMPORT_ERROR = str(e)

try:
    import requests
    from bs4 import BeautifulSoup
    SCRAPER_AVAILABLE = True
except ImportError:
    SCRAPER_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# ==========================================
# 1. 환경 설정 및 상수
# ==========================================
COMMISSION_RATE = 0.00015
SELL_TAX_RATE = 0.0018

UNIVERSE = {
    "삼성전자": "005930",
    "SK하이닉스": "000660",
    "NAVER": "035420",
    "카카오": "035720",
    "LG에너지솔루션": "373220",
    "현대차": "005380",
    "기아": "000270",
    "POSCO홀딩스": "005490",
    "삼성바이오로직스": "207940",
    "셀트리온": "068270",
    "LG화학": "051910",
    "삼성SDI": "006400",
    "현대모비스": "012330",
    "KB금융": "105560",
    "신한지주": "055550",
    "에코프로비엠": "247540",
    "삼양식품": "003230",
    "HD현대중공업": "329180",
    "한화에어로스페이스": "012450",
    "두산에너빌리티": "034020",
}

# 재무제표를 못 가져올 때 쓰는 최후의 폴백 값 (DART 키 없을 때만 사용, 실전 비교엔 부적합)
FALLBACK_FUNDAMENTALS = {
    "삼성전자": {"growth_rate": 20, "debt_ratio": 35},
    "SK하이닉스": {"growth_rate": 25, "debt_ratio": 40},
    "NAVER": {"growth_rate": 12, "debt_ratio": 30},
    "카카오": {"growth_rate": 8, "debt_ratio": 45},
    "LG에너지솔루션": {"growth_rate": 15, "debt_ratio": 90},
    "현대차": {"growth_rate": 15, "debt_ratio": 60},
    "기아": {"growth_rate": 18, "debt_ratio": 55},
    "POSCO홀딩스": {"growth_rate": 10, "debt_ratio": 65},
    "삼성바이오로직스": {"growth_rate": 22, "debt_ratio": 25},
    "셀트리온": {"growth_rate": 14, "debt_ratio": 20},
    "LG화학": {"growth_rate": 9, "debt_ratio": 80},
    "삼성SDI": {"growth_rate": 17, "debt_ratio": 70},
    "현대모비스": {"growth_rate": 11, "debt_ratio": 40},
    "KB금융": {"growth_rate": 7, "debt_ratio": 90},
    "신한지주": {"growth_rate": 7, "debt_ratio": 90},
    "에코프로비엠": {"growth_rate": 10, "debt_ratio": 120},
    "삼양식품": {"growth_rate": 35, "debt_ratio": 45},
    "HD현대중공업": {"growth_rate": 20, "debt_ratio": 100},
    "한화에어로스페이스": {"growth_rate": 30, "debt_ratio": 110},
    "두산에너빌리티": {"growth_rate": 16, "debt_ratio": 95},
}

st.set_page_config(page_title="피터 린치 AI 한국주식 시뮬레이터", layout="wide")


# ==========================================
# 2. 가격 + PER 데이터 엔진 (FinanceDataReader + pykrx)
# ==========================================
class PriceEngine:
    _price_cache = {}   # {symbol: DataFrame(OHLCV)}
    _per_cache = {}      # {symbol: DataFrame(PER 등)}

    @classmethod
    def load_prices(cls, symbol, code, start_date, end_date):
        if symbol not in cls._price_cache:
            df = fdr.DataReader(code, start_date, end_date)
            cls._price_cache[symbol] = df
        return cls._price_cache[symbol]

    @classmethod
    def load_per(cls, symbol, code, start_date, end_date):
        """일자별 실제 PER (pykrx, KRX 산출 기준)"""
        if not PYKRX_AVAILABLE:
            return None
        if symbol not in cls._per_cache:
            fromdate = pd.Timestamp(start_date).strftime("%Y%m%d")
            todate = pd.Timestamp(end_date).strftime("%Y%m%d")
            try:
                df = pykrx_stock.get_market_fundamental_by_date(fromdate, todate, code)
                cls._per_cache[symbol] = df
            except Exception:
                cls._per_cache[symbol] = pd.DataFrame()
        return cls._per_cache[symbol]

    @classmethod
    def get_trading_days(cls, start_date, end_date):
        if FDR_AVAILABLE:
            ref = fdr.DataReader("005930", start_date, end_date)
            return [d.strftime("%Y-%m-%d") for d in ref.index]
        dates = pd.date_range(start=start_date, end=end_date, freq="B")
        return [d.strftime("%Y-%m-%d") for d in dates]

    @classmethod
    def get_price_on(cls, symbol, code, current_date, start_date, end_date):
        df = cls.load_prices(symbol, code, start_date, end_date)
        available = df[df.index <= pd.Timestamp(current_date)]
        if available.empty:
            return None
        return float(available["Close"].iloc[-1])

    @classmethod
    def get_per_on(cls, symbol, code, current_date, start_date, end_date):
        df = cls.load_per(symbol, code, start_date, end_date)
        if df is None or df.empty:
            return None
        available = df[df.index <= pd.Timestamp(current_date)]
        if available.empty:
            return None
        per = float(available["PER"].iloc[-1])
        if per <= 0 or np.isnan(per):
            return None
        return per


# ==========================================
# 3. 재무제표 엔진 (OpenDartReader) - 성장률 / 부채비율
# ==========================================
class FundamentalEngine:
    """
    DART 공시 재무제표를 이용해 '실제 공시 시점' 기준으로만 데이터를 제공한다.
    분기보고서 법정 제출기한(약칭):
        1분기(3월말 결산)   -> 제출기한 약 5/15  (reprt_code 11013)
        반기(6월말 결산)    -> 제출기한 약 8/14  (reprt_code 11012)
        3분기(9월말 결산)   -> 제출기한 약 11/14 (reprt_code 11014)
        사업보고서(12월말)  -> 제출기한 약 3/31  (reprt_code 11011, 전년도 실적)
    실제 제출일은 기업마다 다르지만, 학습 목적상 법정기한을 '공시 가능 시점'으로 근사한다.
    """

    _report_cache = {}  # {(symbol, year, reprt_code): dict or None}

    QUARTER_SCHEDULE = [
        # (기준월일, reprt_code, 회계연도 계산 방식)
        ("05-15", "11013"),  # 1분기
        ("08-14", "11012"),  # 반기
        ("11-14", "11014"),  # 3분기
        ("03-31", "11011"),  # 사업보고서(전년도)
    ]

    def __init__(self, api_key):
        self.api_key = api_key
        self.client = None
        self.init_error = None
        if api_key and DART_LIB_AVAILABLE:
            cleaned_key = api_key.strip()  # 공백/줄바꿈 실수 방지
            try:
                self.client = OpenDartReader(cleaned_key)
                # 초기화만으로는 키 유효성이 검증되지 않으므로, 실제 조회를 한 번 해봐서 확인
                test_df = self.client.finstate("005930", datetime.datetime.now().year - 1, reprt_code="11011")
                if test_df is None or (hasattr(test_df, "empty") and test_df.empty):
                    self.init_error = (
                        "API 키는 형식상 통과했지만 테스트 조회 결과가 비어 있습니다. "
                        "키가 아직 활성화되지 않았거나, 요청 한도를 초과했을 수 있습니다."
                    )
                    self.client = None
            except Exception as e:
                self.init_error = f"{type(e).__name__}: {e}"
                self.client = None
        elif api_key and not DART_LIB_AVAILABLE:
            self.init_error = (
                f"OpenDartReader 패키지 import 실패: {DART_LIB_IMPORT_ERROR}. "
                "requirements.txt에 `OpenDartReader`가 정확히 들어있는지, "
                "Streamlit Cloud의 'Manage app' 로그에서 설치 에러가 없는지 확인하세요."
            )

    def _available_report(self, current_date):
        """current_date 기준으로 이미 공시됐을 것으로 간주할 수 있는 (year, reprt_code) 결정"""
        cd = pd.Timestamp(current_date)
        year = cd.year

        candidates = []
        # 올해 발표분
        candidates.append((pd.Timestamp(f"{year}-05-15"), year, "11013"))
        candidates.append((pd.Timestamp(f"{year}-08-14"), year, "11012"))
        candidates.append((pd.Timestamp(f"{year}-11-14"), year, "11014"))
        # 작년 사업보고서(올해 3/31 공시, 대상연도는 작년)
        candidates.append((pd.Timestamp(f"{year}-03-31"), year - 1, "11011"))
        # 재작년 사업보고서(작년 3/31 이전이면 이것만 유효할 수 있음)
        candidates.append((pd.Timestamp(f"{year - 1}-03-31"), year - 2, "11011"))

        valid = [c for c in candidates if c[0] <= cd]
        if not valid:
            return None
        valid.sort(key=lambda x: x[0])
        _, y, code = valid[-1]
        return y, code

    def get_fundamentals(self, symbol, code, current_date):
        """반환: {"growth_rate": float(%), "debt_ratio": float(%)} 또는 None(조회 실패)"""
        if self.client is None:
            return None

        report = self._available_report(current_date)
        if report is None:
            return None
        year, reprt_code = report

        cache_key = (symbol, year, reprt_code)
        if cache_key in self._report_cache:
            return self._report_cache[cache_key]

        result = None
        try:
            df = self.client.finstate(code, year, reprt_code=reprt_code)
            if df is not None and not df.empty:
                result = self._extract_metrics(df)
        except Exception:
            result = None

        self._report_cache[cache_key] = result
        return result

    @staticmethod
    def _extract_metrics(df):
        """finstate 결과에서 매출액/영업이익 YoY 성장률과 부채비율 계산"""

        def find_amount(keyword):
            rows = df[df["account_nm"].astype(str).str.contains(keyword, na=False)]
            if rows.empty:
                return None, None
            row = rows.iloc[0]
            try:
                thstrm = float(str(row.get("thstrm_amount", "0")).replace(",", ""))
                frmtrm = float(str(row.get("frmtrm_amount", "0")).replace(",", ""))
                return thstrm, frmtrm
            except (ValueError, TypeError):
                return None, None

        op_thstrm, op_frmtrm = find_amount("영업이익")
        revenue_thstrm, revenue_frmtrm = find_amount("매출액")
        debt_thstrm, _ = find_amount("부채총계")
        equity_thstrm, _ = find_amount("자본총계")

        growth_rate = None
        if op_thstrm is not None and op_frmtrm and op_frmtrm != 0:
            growth_rate = (op_thstrm - op_frmtrm) / abs(op_frmtrm) * 100
        elif revenue_thstrm is not None and revenue_frmtrm and revenue_frmtrm != 0:
            growth_rate = (revenue_thstrm - revenue_frmtrm) / abs(revenue_frmtrm) * 100

        debt_ratio = None
        if debt_thstrm is not None and equity_thstrm and equity_thstrm != 0:
            debt_ratio = debt_thstrm / equity_thstrm * 100

        if growth_rate is None or debt_ratio is None:
            return None

        return {"growth_rate": round(growth_rate, 2), "debt_ratio": round(debt_ratio, 2)}


# ==========================================
# 4. 뉴스 엔진 (네이버 금융 종목뉴스 크롤링)
# ==========================================
class NewsEngine:
    """
    finance.naver.com/item/news_news.naver?code={종목코드}&page={n}
    페이지를 뒤로 넘기며 start_date 이전까지 수집한다.
    페이지 구조가 바뀌면 셀렉터를 다시 확인해야 한다.
    """

    _news_cache = {}  # {symbol: {"YYYY-MM-DD": [title, ...]}}

    HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    @classmethod
    def _scrape(cls, code, start_date, end_date, max_pages=15, delay_sec=0.25):
        news_by_date = {}
        start_ts = pd.Timestamp(start_date)

        for page in range(1, max_pages + 1):
            url = f"https://finance.naver.com/item/news_news.naver?code={code}&page={page}"
            try:
                res = requests.get(url, headers=cls.HEADERS, timeout=5)
                res.encoding = "euc-kr"
            except Exception:
                break

            soup = BeautifulSoup(res.text, "html.parser")
            rows = soup.select("table.type5 tr")

            page_had_rows = False
            oldest_on_page = None

            for row in rows:
                title_tag = row.select_one("td.title a")
                date_tag = row.select_one("td.date")
                if not title_tag or not date_tag:
                    continue

                title = title_tag.get_text(strip=True)
                date_str = date_tag.get_text(strip=True)
                try:
                    dt = datetime.datetime.strptime(date_str, "%Y.%m.%d %H:%M")
                except ValueError:
                    continue

                page_had_rows = True
                key = dt.strftime("%Y-%m-%d")
                news_by_date.setdefault(key, [])
                if title not in news_by_date[key]:
                    news_by_date[key].append(title)

                if oldest_on_page is None or dt < oldest_on_page:
                    oldest_on_page = dt

            if not page_had_rows:
                break
            if oldest_on_page is not None and oldest_on_page < start_ts:
                break

            time.sleep(delay_sec)  # 서버 부담 방지

        return news_by_date

    @classmethod
    def load_news(cls, symbol, code, start_date, end_date):
        if not SCRAPER_AVAILABLE:
            return {}
        if symbol not in cls._news_cache:
            try:
                cls._news_cache[symbol] = cls._scrape(code, start_date, end_date)
            except Exception:
                cls._news_cache[symbol] = {}
        return cls._news_cache[symbol]

    @classmethod
    def get_news_on(cls, symbol, code, current_date, start_date, end_date):
        news_by_date = cls.load_news(symbol, code, start_date, end_date)
        titles = news_by_date.get(current_date, [])
        if not titles:
            return f"[{current_date}] {symbol} 특이 뉴스 없음."
        headline = " / ".join(titles[:2])  # 당일 기사 상위 2개만 사용
        return f"[{current_date}] {symbol} {headline}"


# ==========================================
# 5. 통합 시장 데이터 엔진
# ==========================================
class StockDataEngine:
    @staticmethod
    def fetch_market_snapshot(current_date, start_date, end_date, fundamental_engine):
        snapshot = {}
        for symbol, code in UNIVERSE.items():
            if FDR_AVAILABLE:
                price = PriceEngine.get_price_on(symbol, code, current_date, start_date, end_date)
                if price is None:
                    continue
            else:
                price = 100000 + np.random.randint(-1000, 1000)  # FDR 미설치 시 폴백(테스트용)

            if PYKRX_AVAILABLE:
                per = PriceEngine.get_per_on(symbol, code, current_date, start_date, end_date)
            else:
                per = None
            if per is None:
                per = 15.0  # PER 조회 실패 시 임시값 (가급적 pykrx 설치 권장)

            fundamentals = fundamental_engine.get_fundamentals(symbol, code, current_date)
            if fundamentals is None:
                # DART 조회 실패 시 폴백 (실전 비교용으로는 부정확할 수 있음을 UI에서 경고)
                fundamentals = FALLBACK_FUNDAMENTALS[symbol]

            snapshot[symbol] = {
                "price": price,
                "per": per,
                "growth_rate": fundamentals["growth_rate"],
                "debt_ratio": fundamentals["debt_ratio"],
            }
        return snapshot

    @staticmethod
    def fetch_daily_news(current_date, symbol):
        code = UNIVERSE[symbol]
        return NewsEngine.get_news_on(symbol, code, current_date, st.session_state["start_date"], st.session_state["end_date"])


# ==========================================
# 6. 피터 린치 AI 엔진
# ==========================================
class PeterLynchAI:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key) if (api_key and OPENAI_AVAILABLE) else None

    def screen_quant(self, market_data, peg_screen_max=1.2, debt_max=100):
        candidates = {}
        for symbol, metrics in market_data.items():
            growth = metrics["growth_rate"]
            per = metrics["per"]
            debt = metrics["debt_ratio"]
            if growth is None or growth <= 0:
                continue
            peg = per / growth
            if peg <= peg_screen_max and debt < debt_max:
                candidates[symbol] = {**metrics, "peg": round(peg, 2)}
        return candidates

    def analyze_and_decide(self, current_date, symbol, metrics, news, peg_buy=1.0, peg_sell=1.3):
        if self.client:
            return self._analyze_with_llm(current_date, symbol, metrics, news)
        return self._analyze_rule_based(metrics, news, peg_buy, peg_sell)

    def _analyze_rule_based(self, metrics, news, peg_buy=1.0, peg_sell=1.3):
        peg = metrics["peg"]
        debt = metrics["debt_ratio"]
        has_positive = ("수출" in news) or ("호조" in news) or ("증가" in news) or ("확대" in news) \
            or ("상승" in news) or ("최대" in news) or ("성장" in news)
        has_negative = ("감소" in news) or ("우려" in news) or ("하락" in news) or ("부진" in news) \
            or ("리콜" in news) or ("소송" in news) or ("적자" in news)

        # 1순위: 뉴스 촉매 + PEG 조건이 맞으면 매수/매도
        if peg <= peg_buy and has_positive:
            action, ratio = "BUY", 0.4
            reason = f"PEG {peg}로 저평가 구간이며 긍정적 뉴스('{news[:40]}...') 확인."
        elif peg > peg_sell and has_negative:
            action, ratio = "SELL", 1.0
            reason = f"PEG({peg}) 고평가 진입 + 부정적 뉴스('{news[:40]}...')로 위험 관리 매도."
        # 2순위: 뉴스 촉매가 없어도 극단적인 저평가/재무 위험이면 판단
        elif peg <= peg_buy * 0.6:
            action, ratio = "BUY", 0.25
            reason = f"뉴스 촉매는 약하지만 PEG {peg}가 뚜렷한 저평가 구간이라 소액 매수."
        elif debt >= 150:
            action, ratio = "SELL", 1.0
            reason = f"부채비율 {debt}%로 재무 위험이 높아 위험 관리 매도."
        else:
            action, ratio = "HOLD", 0
            reason = "펀더멘털은 양호하나 결정적 촉매 부족으로 관망."
        return action, ratio, reason

    def _analyze_with_llm(self, current_date, symbol, metrics, news):
        system_prompt = (
            "당신은 피터 린치의 GARP 투자 철학을 따르는 애널리스트입니다. "
            "PEG, 부채비율, 뉴스를 바탕으로 BUY/SELL/HOLD를 결정하고 간결한 한국어 사유를 제시하세요. "
            "반드시 아래 JSON 형식으로만 답하세요: "
            '{"action": "BUY|SELL|HOLD", "ratio": 0.0~1.0, "reason": "판단 사유"}'
        )
        user_prompt = (
            f"날짜: {current_date}\n종목: {symbol}\nPEG: {metrics['peg']}\nPER: {metrics['per']}\n"
            f"이익성장률: {metrics['growth_rate']}%\n부채비율: {metrics['debt_ratio']}%\n오늘 뉴스: {news}"
        )
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            content = response.choices[0].message.content.strip()
            content = content.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(content)
            action = parsed.get("action", "HOLD")
            if action not in ("BUY", "SELL", "HOLD"):
                action = "HOLD"
            return action, float(parsed.get("ratio", 0)), f"[LLM] {parsed.get('reason', '')}"
        except Exception as e:
            action, ratio, reason = self._analyze_rule_based(metrics, news)
            return action, ratio, f"[LLM 호출 실패({e}) → 룰베이스 폴백] {reason}"


# ==========================================
# 7. 성과 지표
# ==========================================
def calculate_performance_metrics(df_history, seed_money):
    df = df_history.copy()
    df["cummax"] = df["total_asset"].cummax()
    df["drawdown"] = (df["total_asset"] - df["cummax"]) / df["cummax"]
    mdd = df["drawdown"].min() * 100
    df["daily_return"] = df["total_asset"].pct_change()
    volatility = df["daily_return"].std() * np.sqrt(252) * 100
    final_asset = df["total_asset"].iloc[-1]
    total_return = ((final_asset - seed_money) / seed_money) * 100
    return {"final_asset": final_asset, "total_return": total_return,
            "mdd": mdd, "volatility": volatility, "df": df}


# ==========================================
# 8. Streamlit UI
# ==========================================
st.title("📈 피터 린치 AI 한국주식 백테스팅")
st.caption("과거 특정 기간의 실제 KRX 종가/PER + DART 공시 재무제표 + 네이버 금융 실제 뉴스를 바탕으로, "
           "그 시점에 AI가 어떻게 판단했을지 되짚어보는 백테스팅 도구입니다.")

missing = []
if not FDR_AVAILABLE:
    missing.append("finance-datareader (가격)")
if not PYKRX_AVAILABLE:
    missing.append("pykrx (PER)")
if not DART_LIB_AVAILABLE:
    missing.append("OpenDartReader (재무제표)")
if not SCRAPER_AVAILABLE:
    missing.append("requests, beautifulsoup4 (뉴스 크롤링)")
if missing:
    st.warning("⚠️ 다음 패키지가 없어 일부 기능이 임시 데이터로 대체됩니다: " + ", ".join(missing) +
               "\n실전 비교를 위해서는 반드시 설치 후 실행하세요.")

with st.sidebar:
    st.header("⚙️ 투자 설정")
    seed_money = st.number_input("초기 시드머니 (원)", min_value=100000, max_value=100000000, value=1000000, step=100000)

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("시작일", datetime.date(2024, 4, 1))
    with col2:
        end_date = st.date_input("종료일", datetime.date(2024, 5, 31))
    st.caption("⏳ 이 도구는 과거 데이터만 사용하는 백테스팅 도구입니다. 종료일은 오늘(2026-08-09)보다 이후일 수 없어요.")

    st.divider()
    st.subheader("데이터 연동 키")
    dart_api_key = st.text_input(
        "DART API Key (무료)", type="password",
        help="https://opendart.fss.or.kr 에서 무료 발급. 없으면 성장률/부채비율이 임시값으로 대체됩니다."
    )
    openai_api_key = st.text_input(
        "OpenAI API Key (선택)", type="password",
        help="입력 시 실제 LLM이 매매를 판단합니다. 미입력 시 룰베이스 모드로 작동합니다."
    )

    st.divider()
    st.subheader("매매 기준 조절 (룰베이스 모드)")
    peg_screen_max = st.slider("1차 스크리닝 PEG 상한", 0.5, 3.0, 1.2, 0.1,
                                help="이 값보다 PEG가 크면 애초에 후보에서 제외됩니다.")
    peg_buy = st.slider("매수 PEG 기준", 0.3, 2.0, 1.0, 0.1,
                         help="이 값 이하 + 긍정 뉴스가 있으면 매수. 값을 높일수록 매수가 잦아집니다.")
    peg_sell = st.slider("매도 PEG 기준", 0.5, 2.5, 1.3, 0.1,
                          help="이 값보다 크고 부정 뉴스가 있으면 매도.")
    debt_max = st.slider("스크리닝 부채비율 상한 (%)", 50, 300, 100, 10)

    st.divider()
    st.subheader("손절/익절 규칙 (거래 활성화)")
    st.caption("PEG·뉴스와 무관하게, 매수가 대비 수익률이 이 범위를 벗어나면 기계적으로 매도합니다. "
               "감정을 배제한 원칙 매매를 흉내내는 핵심 장치이며, 거래 빈도를 크게 늘려줍니다.")
    stop_loss_pct = st.slider("손절 기준 (%)", -30, -1, -8, 1)
    take_profit_pct = st.slider("익절 기준 (%)", 3, 50, 15, 1)
    max_positions = st.slider("동시 보유 최대 종목 수", 1, 15, 6, 1,
                               help="너무 크면 종목당 투자금이 작아지고, 너무 작으면 후보가 있어도 매수를 못 합니다.")

    st.divider()
    debug_mode = st.checkbox("🔍 디버그 모드 (일별 판단 근거 전부 보기)", value=False)

    run_btn = st.button("🚀 시뮬레이션 실행", use_container_width=True)

if start_date >= end_date:
    st.error("오류: 종료일은 시작일보다 이후여야 합니다.")
    st.stop()

TODAY = datetime.date(2026, 8, 9)
if end_date > TODAY:
    st.error(f"오류: 종료일은 오늘({TODAY}) 이후일 수 없습니다. 이 도구는 과거 데이터만 조회 가능합니다.")
    st.stop()

st.session_state["start_date"] = start_date
st.session_state["end_date"] = end_date


# ==========================================
# 9. 실행 로직
# ==========================================
if run_btn:
    fundamental_engine = FundamentalEngine(api_key=dart_api_key)
    if dart_api_key and fundamental_engine.client is None:
        error_detail = fundamental_engine.init_error or "알 수 없는 오류"
        st.error(f"DART API 연동 실패: {error_detail}\n\n임시 재무 데이터로 진행합니다.")

    with st.spinner("실제 시세/재무/뉴스 데이터를 불러오는 중입니다 (뉴스 크롤링은 다소 시간이 걸릴 수 있어요)..."):
        trading_days = PriceEngine.get_trading_days(start_date, end_date)
        # 뉴스는 종목별로 한 번만 미리 긁어서 캐싱해둔다
        for symbol, code in UNIVERSE.items():
            NewsEngine.load_news(symbol, code, start_date, end_date)

    ai = PeterLynchAI(api_key=openai_api_key if OPENAI_AVAILABLE else None)

    cash = seed_money
    portfolio = {}
    history = []
    trade_logs = []
    debug_logs = []

    progress_bar = st.progress(0)

    for idx, current_date in enumerate(trading_days):
        market_data = StockDataEngine.fetch_market_snapshot(current_date, start_date, end_date, fundamental_engine)
        candidates = ai.screen_quant(market_data, peg_screen_max=peg_screen_max, debt_max=debt_max)

        # 보유 종목 매도 검토
        for symbol in list(portfolio.keys()):
            if symbol not in market_data:
                continue
            metrics = market_data[symbol]
            if metrics["growth_rate"] and metrics["growth_rate"] > 0:
                metrics["peg"] = round(metrics["per"] / metrics["growth_rate"], 2)
            else:
                metrics["peg"] = 999

            # 1순위: 손절/익절 규칙 (PEG·뉴스와 무관하게 기계적으로 적용, 거래 활성화 핵심 장치)
            avg_price = portfolio[symbol]["avg_price"]
            current_return_pct = (metrics["price"] - avg_price) / avg_price * 100
            if current_return_pct <= stop_loss_pct:
                action = "SELL"
                reason = f"손절 기준({stop_loss_pct}%) 도달 (현재 수익률 {current_return_pct:.1f}%) → 규칙 매도"
                news = "규칙 기반 판단 (뉴스 미조회)"
            elif current_return_pct >= take_profit_pct:
                action = "SELL"
                reason = f"익절 기준({take_profit_pct}%) 도달 (현재 수익률 {current_return_pct:.1f}%) → 규칙 매도"
                news = "규칙 기반 판단 (뉴스 미조회)"
            else:
                news = StockDataEngine.fetch_daily_news(current_date, symbol)
                action, ratio, reason = ai.analyze_and_decide(current_date, symbol, metrics, news, peg_buy, peg_sell)

            if debug_mode:
                debug_logs.append({"date": current_date, "symbol": symbol, "peg": metrics["peg"],
                                    "debt": metrics["debt_ratio"], "return_pct": round(current_return_pct, 2),
                                    "in_portfolio": True, "action": action, "news": news})
            if action == "SELL" and portfolio[symbol]["shares"] > 0:
                shares_to_sell = portfolio[symbol]["shares"]
                sell_price = metrics["price"]
                gross = shares_to_sell * sell_price
                fee = gross * COMMISSION_RATE
                tax = gross * SELL_TAX_RATE
                cash += gross - fee - tax
                trade_logs.append({"date": current_date, "symbol": symbol, "action": "매도",
                                    "price": sell_price, "shares": shares_to_sell, "reason": reason})
                del portfolio[symbol]

        # 신규 매수 검토
        for symbol, metrics in candidates.items():
            if len(portfolio) >= max_positions:
                break  # 최대 보유 종목 수 도달 시 추가 매수 중단
            if symbol not in portfolio and cash > 50000:
                news = StockDataEngine.fetch_daily_news(current_date, symbol)
                action, ratio, reason = ai.analyze_and_decide(current_date, symbol, metrics, news, peg_buy, peg_sell)
                if debug_mode:
                    debug_logs.append({"date": current_date, "symbol": symbol, "peg": metrics["peg"],
                                        "debt": metrics["debt_ratio"], "return_pct": None,
                                        "in_portfolio": False, "action": action, "news": news})
                if action == "BUY":
                    # 남은 보유 슬롯 수에 맞춰 투자 비중을 배분 (한 종목에 몰리지 않도록)
                    remaining_slots = max(max_positions - len(portfolio), 1)
                    target_budget = min(cash * ratio, cash / remaining_slots)
                    buy_price = metrics["price"]
                    max_shares = int(target_budget // (buy_price * (1 + COMMISSION_RATE)))
                    if max_shares > 0:
                        cash -= max_shares * buy_price * (1 + COMMISSION_RATE)
                        portfolio[symbol] = {"shares": max_shares, "avg_price": buy_price}
                        trade_logs.append({"date": current_date, "symbol": symbol, "action": "매수",
                                            "price": buy_price, "shares": max_shares, "reason": reason})

        portfolio_eval = sum(info["shares"] * market_data[sym]["price"]
                              for sym, info in portfolio.items() if sym in market_data)
        total_asset = cash + portfolio_eval
        history.append({"date": current_date, "total_asset": total_asset, "cash": cash, "portfolio_eval": portfolio_eval})
        progress_bar.progress((idx + 1) / len(trading_days))

    # ==========================================
    # 10. 결과 출력
    # ==========================================
    df_history = pd.DataFrame(history)
    result = calculate_performance_metrics(df_history, seed_money)

    st.subheader("📊 시뮬레이션 결과 요약")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("최종 자산", f"{int(result['final_asset']):,} 원")
    c2.metric("총 수익률", f"{result['total_return']:.2f}%", delta=f"{result['total_return']:.2f}%")
    c3.metric("최대낙폭 (MDD)", f"{result['mdd']:.2f}%")
    c4.metric("연환산 변동성", f"{result['volatility']:.2f}%")
    st.caption(f"총 매매 횟수: {len(trade_logs)}회")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_history["date"], y=df_history["total_asset"], mode="lines", name="총 자산"))
    fig.add_hline(y=seed_money, line_dash="dash", line_color="gray", annotation_text="원금")
    fig.update_layout(title="일별 자산 추이", xaxis_title="날짜", yaxis_title="자산 (원)")
    st.plotly_chart(fig, use_container_width=True)

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=result["df"]["date"], y=result["df"]["drawdown"] * 100,
                                  mode="lines", fill="tozeroy", name="낙폭(%)", line_color="crimson"))
    fig_dd.update_layout(title="낙폭(Drawdown) 추이", xaxis_title="날짜", yaxis_title="낙폭 (%)")
    st.plotly_chart(fig_dd, use_container_width=True)

    st.subheader("📝 AI 매매 타임라인 및 근거 (실제 뉴스 기반)")
    if trade_logs:
        for log in reversed(trade_logs):
            with st.expander(f"[{log['date']}] {log['symbol']} {log['action']} ({log['shares']}주 @ {log['price']:,}원)"):
                st.write(f"**매매 사유:** {log['reason']}")
    else:
        st.info("해당 기간 동안 매매 조건을 충족하는 기회가 없었습니다.")

    st.subheader("💾 결과 내보내기")
    csv = df_history.to_csv(index=False).encode("utf-8-sig")
    st.download_button("자산 추이 CSV 다운로드", data=csv, file_name="ai_asset_history.csv", mime="text/csv")

    if debug_mode and debug_logs:
        st.subheader("🔍 디버그: 일별 판단 근거 전체")
        df_debug = pd.DataFrame(debug_logs)

        no_news_ratio = (df_debug["news"].str.contains("특이 뉴스 없음")).mean() * 100
        st.caption(
            f"전체 판단 중 '특이 뉴스 없음'으로 처리된 비율: {no_news_ratio:.1f}% "
            "(이 비율이 90% 이상이면 뉴스 크롤링이 제대로 안 되고 있을 가능성이 높습니다)"
        )
        action_counts = df_debug["action"].value_counts()
        st.write("판단 결과 분포:", action_counts.to_dict())

        st.dataframe(df_debug, use_container_width=True, height=400)
