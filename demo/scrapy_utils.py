import requests
from SPARQLWrapper import SPARQLWrapper, JSON

"""
    1. 使用SPARKQL查询接口 DBPedia
"""


def get_dbpedia_abstract_sqarkql(ent, lang='en'):
    endpoint_url = 'http://dbpedia.org/sparql'
    sparql = SPARQLWrapper(endpoint_url)

    query = f"""
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX dbr: <http://dbpedia.org/resource/>
    SELECT ?abstract WHERE {{ 
        dbr:{ent} dbo:abstract ?abstract .
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
            return f'查询失败'
    except Exception as e:
        print(e)
        return f'查询失败'


"""
    2. 使用REST查询接口 DBPedia
"""


def get_dbpedia_abstract_json(ent, lang='en'):
    url = f'https://dbpedia.org/data/{ent}.json'
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
    res = get_dbpedia_abstract_json('Logic')
    print(res)
