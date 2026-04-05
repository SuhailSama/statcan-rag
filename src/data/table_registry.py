"""
Curated registry of high-value StatCan tables.

Instead of searching StatCan's API cold (which is slow), we maintain
a hand-picked list of the most useful tables with plain-English descriptions
and keyword tags. The RAG engine uses this for fast, relevant lookup.
"""

from .models import StatCanTable

# fmt: off
_REGISTRY: list[dict] = [
    {
        "pid": "14-10-0287-01",
        "title": "Labour force characteristics by province, monthly, seasonally adjusted",
        "description": "Canada's monthly Labour Force Survey — employment, unemployment rate, participation rate, and hours worked by province.",
        "keywords": ["employment", "unemployment", "jobs", "labour force", "participation rate", "LFS", "workforce"],
        "frequency": "monthly",
        "category": "Labour",
    },
    {
        "pid": "36-10-0104-01",
        "title": "Gross domestic product (GDP) at basic prices, by industry",
        "description": "Monthly GDP at basic prices by industry in Canada, chained dollars.",
        "keywords": ["GDP", "gross domestic product", "economic output", "industry", "growth"],
        "frequency": "monthly",
        "category": "Economy",
    },
    {
        "pid": "18-10-0004-01",
        "title": "Consumer Price Index, monthly, not seasonally adjusted",
        "description": "Canada's CPI — measures changes in prices paid by consumers for goods and services (inflation indicator).",
        "keywords": ["CPI", "consumer price index", "inflation", "prices", "cost of living"],
        "frequency": "monthly",
        "category": "Prices",
    },
    {
        "pid": "34-10-0145-01",
        "title": "Building permits, by type of structure and type of work",
        "description": "Monthly building permit values and units by residential, commercial, industrial sector.",
        "keywords": ["building permits", "construction", "housing starts", "real estate", "development"],
        "frequency": "monthly",
        "category": "Construction",
    },
    {
        "pid": "17-10-0005-01",
        "title": "Population estimates on July 1st, by age and sex",
        "description": "Annual population estimates for Canada and provinces/territories by age group and sex.",
        "keywords": ["population", "demographics", "age", "sex", "census", "growth"],
        "frequency": "annual",
        "category": "Demographics",
    },
    {
        "pid": "12-10-0121-01",
        "title": "International merchandise trade by commodity",
        "description": "Monthly exports and imports of goods by commodity type — Canada's trade balance.",
        "keywords": ["trade", "exports", "imports", "merchandise", "international", "balance of trade"],
        "frequency": "monthly",
        "category": "Trade",
    },
    {
        "pid": "36-10-0434-01",
        "title": "Gross domestic product, expenditure-based, quarterly",
        "description": "Quarterly GDP measured from expenditure side — consumption, investment, government, net exports.",
        "keywords": ["GDP", "expenditure", "consumption", "investment", "government spending", "quarterly"],
        "frequency": "quarterly",
        "category": "Economy",
    },
    {
        "pid": "11-10-0239-01",
        "title": "Income of individuals by age group, sex and income source",
        "description": "Annual income statistics for Canadians including wages, transfers, and total income by demographics.",
        "keywords": ["income", "wages", "earnings", "salary", "poverty", "inequality", "individuals"],
        "frequency": "annual",
        "category": "Income",
    },
    {
        "pid": "46-10-0046-01",
        "title": "New housing price index, monthly",
        "description": "Monthly index of prices of new residential houses sold by contractors in Canada.",
        "keywords": ["housing prices", "new housing", "real estate", "price index", "home prices", "NHPI"],
        "frequency": "monthly",
        "category": "Housing",
    },
    {
        "pid": "14-10-0020-01",
        "title": "Unemployment rate by province",
        "description": "Monthly unemployment rates for each Canadian province and territory.",
        "keywords": ["unemployment", "jobless rate", "province", "regional", "labour market"],
        "frequency": "monthly",
        "category": "Labour",
    },
    {
        "pid": "71-607-X2018011",
        "title": "Job vacancy and wage survey",
        "description": "Quarterly job vacancy rates and average offered wages by industry and province.",
        "keywords": ["job vacancy", "wages", "labour shortage", "hiring", "openings"],
        "frequency": "quarterly",
        "category": "Labour",
    },
    {
        "pid": "36-10-0008-01",
        "title": "Gross domestic product (GDP), income-based, quarterly",
        "description": "Quarterly GDP from income side — compensation of employees, gross operating surplus, taxes.",
        "keywords": ["GDP", "income", "compensation", "corporate profits", "quarterly"],
        "frequency": "quarterly",
        "category": "Economy",
    },
    {
        "pid": "18-10-0006-01",
        "title": "Consumer Price Index, annual average, not seasonally adjusted",
        "description": "Annual average CPI by province and major cities. Useful for year-over-year inflation comparisons.",
        "keywords": ["CPI", "inflation", "annual", "provincial", "cities", "cost of living"],
        "frequency": "annual",
        "category": "Prices",
    },
    {
        "pid": "32-10-0077-01",
        "title": "Farm cash receipts, annual",
        "description": "Annual cash receipts from Canadian agricultural operations by province and commodity.",
        "keywords": ["agriculture", "farming", "crops", "livestock", "farm income"],
        "frequency": "annual",
        "category": "Agriculture",
    },
    {
        "pid": "25-10-0015-01",
        "title": "Electric power generation, monthly",
        "description": "Monthly electricity generation by source (hydro, nuclear, wind, solar, fossil fuels) by province.",
        "keywords": ["electricity", "energy", "power generation", "hydro", "nuclear", "renewable", "wind", "solar"],
        "frequency": "monthly",
        "category": "Energy",
    },
    {
        "pid": "23-10-0066-01",
        "title": "Air carrier traffic at Canadian airports, monthly",
        "description": "Monthly statistics on passenger volumes, flights, and cargo at major Canadian airports.",
        "keywords": ["air travel", "airports", "passengers", "aviation", "flights", "airlines"],
        "frequency": "monthly",
        "category": "Transportation",
    },
    {
        "pid": "20-10-0008-01",
        "title": "Retail trade sales by province and territory, monthly",
        "description": "Monthly retail sales volumes by province — a key indicator of consumer spending.",
        "keywords": ["retail", "sales", "consumer spending", "shopping", "stores", "commerce"],
        "frequency": "monthly",
        "category": "Retail",
    },
    {
        "pid": "27-10-0273-01",
        "title": "Research and development expenditures, by sector",
        "description": "Annual R&D spending by businesses, government, universities, and non-profits in Canada.",
        "keywords": ["research", "R&D", "innovation", "science", "technology", "development spending"],
        "frequency": "annual",
        "category": "Innovation",
    },
    {
        "pid": "43-10-0024-01",
        "title": "Cybersecurity statistics, annual",
        "description": "Annual statistics on cybersecurity incidents, preparedness, and spending by Canadian businesses.",
        "keywords": ["cybersecurity", "data breach", "hacking", "IT security", "digital"],
        "frequency": "annual",
        "category": "Technology",
    },
    {
        "pid": "17-10-0008-01",
        "title": "Estimates of the components of demographic growth, annual",
        "description": "Annual births, deaths, immigration, emigration, and net migration driving Canada's population change.",
        "keywords": ["immigration", "births", "deaths", "migration", "population growth", "demographics"],
        "frequency": "annual",
        "category": "Demographics",
    },
    {
        "pid": "98-10-0356-01",
        "title": "Language spoken most often at home by age",
        "description": "Census data on languages spoken at home by Canadians, by age and geography.",
        "keywords": ["language", "French", "English", "bilingual", "census", "official languages"],
        "frequency": "annual",
        "category": "Demographics",
    },
    {
        "pid": "14-10-0023-01",
        "title": "Employment by industry, monthly",
        "description": "Monthly employment counts broken down by major industry sector across Canada.",
        "keywords": ["employment", "industry", "jobs", "sector", "workforce", "manufacturing", "services"],
        "frequency": "monthly",
        "category": "Labour",
    },
    {
        "pid": "36-10-0577-01",
        "title": "Business conditions survey, quarterly",
        "description": "Quarterly survey of Canadian business confidence, expectations for sales, hiring, and investment.",
        "keywords": ["business confidence", "outlook", "expectations", "investment", "survey"],
        "frequency": "quarterly",
        "category": "Business",
    },
    {
        "pid": "10-10-0122-01",
        "title": "Canada's international investment position, quarterly",
        "description": "Quarterly data on Canada's assets and liabilities with the rest of the world (foreign investment).",
        "keywords": ["foreign investment", "FDI", "international", "assets", "liabilities", "capital"],
        "frequency": "quarterly",
        "category": "Trade",
    },
    {
        "pid": "38-10-0232-01",
        "title": "Greenhouse gas emissions, by province and economic sector",
        "description": "Annual greenhouse gas emissions (CO2 equivalent) by province and sector.",
        "keywords": ["greenhouse gas", "emissions", "CO2", "climate change", "environment", "carbon"],
        "frequency": "annual",
        "category": "Environment",
    },
]
# fmt: on


def _make_table(d: dict) -> StatCanTable:
    pid_clean = d["pid"].replace("-", "")
    return StatCanTable(
        pid=d["pid"],
        title=d["title"],
        description=d["description"],
        keywords=d["keywords"],
        frequency=d["frequency"],
        category=d["category"],
        url=f"https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid={pid_clean}",
    )


class TableRegistry:
    """In-memory catalog of curated StatCan tables with keyword search."""

    def __init__(self):
        self._tables = [_make_table(d) for d in _REGISTRY]

    def get_all(self) -> list[StatCanTable]:
        return self._tables

    def search(self, query: str, top_k: int = 5) -> list[StatCanTable]:
        """
        Simple keyword search over table metadata.
        Scores each table by how many query words appear in its text fields,
        then returns the top matches.
        """
        q_words = query.lower().split()
        scored: list[tuple[int, StatCanTable]] = []

        for table in self._tables:
            haystack = " ".join([
                table.title,
                table.description,
                " ".join(table.keywords),
                table.category,
                table.frequency,
            ]).lower()
            score = sum(haystack.count(w) for w in q_words)
            if score > 0:
                scored.append((score, table))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:top_k]]

    def get_by_pid(self, pid: str) -> StatCanTable | None:
        for t in self._tables:
            if t.pid == pid:
                return t
        return None
