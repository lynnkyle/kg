import requests
from SPARQLWrapper import SPARQLWrapper, JSON
from urllib.parse import quote

"""
    1. 使用SPARKQL查询接口 DBPedia
"""


def get_dbpedia_ent_abstract_sparql(ent, lang='en'):
    endpoint_url = 'http://dbpedia.org/sparql'
    sparql = SPARQLWrapper(endpoint_url)

    full_uri = f"http://dbpedia.org/resource/{ent}"

    # 使用完整 URI，避免语法错误
    query = f"""
    PREFIX dbo: <http://dbpedia.org/ontology/>
    SELECT ?abstract WHERE {{
        <{full_uri}> dbo:abstract ?abstract .
        FILTER (lang(?abstract) = '{lang}')
    }}
    """

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    try:
        results = sparql.query().convert()
        bindings = results.get('results', {}).get('bindings', [])
        if bindings:
            return bindings[0]['abstract']['value']
        else:
            return None
    except Exception as e:
        print(f"SPARQL query error: {e}")
        return None


"""
    2. 使用REST查询接口 DBPedia
"""


def get_dbpedia_abstract_json(ent, lang='en'):
    url = f'http://dbpedia.org/data/{ent}.json'
    response = requests.get(url)
    if response.status_code != 200:
        return None

    data = response.json()
    subject_uri = f'http://dbpedia.org/resource/{ent}'

    abstracts = data.get(subject_uri, {}).get('http://dbpedia.org/ontology/abstract', [])
    for item in abstracts:
        if item.get('lang', {}) == lang:
            return item.get('value')
    return None


if __name__ == '__main__':
    res = get_dbpedia_ent_abstract_sparql('Montgomery,_Alabama')
    print(res)
