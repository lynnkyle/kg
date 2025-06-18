import os
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from SPARQLWrapper import SPARQLWrapper, JSON
from urllib.parse import quote


def load_entity(data_dir):
    path = os.path.join(data_dir, 'entity2id.txt')
    ent2id = {}
    id2ent = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        ent_num = lines[0]
        for line in lines[1:]:
            if not line:
                continue
            url, _ = line.strip().split(' ')
            ent2id[url] = _
            id2ent.append(url)
    return ent_num, ent2id, id2ent


def load_relation(data_dir):
    path = os.path.join(data_dir, 'relation2id.txt')
    rel2id = {}
    id2rel = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        rel_num = lines[0]
        for line in lines[1:]:
            if not line:
                continue
            if line is None or line == '\n':
                continue
            url, _ = line.strip().split(' ')
            rel2id[url] = _
            id2rel.append(url)
    return rel_num, rel2id, id2rel


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
            return ""
    except Exception as e:
        print(f"SPARQL query error: {e}")
        return ""


def get_dbpedia_rel_desc_json(name, lang='en'):
    pass


def save_ent2desc(data_dir):
    path = os.path.join(data_dir, 'entity.json')
    ent_num, ent2id, id2ent = load_entity(data_dir)
    res_dict = {}

    def process(ent):
        ent_name = ent.strip('<>').split('/')[-1]
        desc = get_dbpedia_ent_abstract_sparql(ent_name, lang='en')
        return ent, {'name': ent_name, 'desc': desc}

    # 开启线程池
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process, ent): ent for ent in id2ent}
        for future in as_completed(futures):
            ent, ent_dict = future.result()
            res_dict[ent] = ent_dict

    # 写入 JSON 文件
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(res_dict, f, ensure_ascii=False, indent=2)


def save_rel2desc(data_dir):
    path = os.path.join(data_dir, 'relation.json')
    rel_num, rel2id, id2rel = load_relation(data_dir)
    res_dict = {}
    for rel in id2rel:
        rel_name = rel.strip('<>').split('/')[-1]
        rel_dict = {'name': rel_name, 'desc': ''}
        res_dict[rel] = rel_dict
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(res_dict, f, ensure_ascii=False, indent=2)


def save_triple2id():
    pass


def save_triple2text(data_dir, in_file, out_file):
    ent_num, ent2id, id2ent = load_entity(data_dir)
    rel_num, rel2id, id2rel = load_relation(data_dir)
    read_path = os.path.join(data_dir, in_file)
    write_path = os.path.join(data_dir, out_file)

    def save_triple(read_path, write_path):
        with open(read_path, 'r', encoding='utf-8') as f1, open(write_path, 'w', encoding='utf-8') as f2:
            for line in f1.readlines():
                h, r, t = line.strip().split('\t')
                f2.write(f'{id2ent[int(h)]}\t{id2rel[int(r)]}\t{id2ent[int(t)]}\n')

    save_triple(read_path, write_path)


if __name__ == '__main__':
    save_triple2text('DB15K', 'train2id.txt', 'train.txt')
    save_triple2text('DB15K', 'test2id.txt', 'test.txt')
    save_triple2text('DB15K', 'valid2id.txt', 'valid.txt')
    save_ent2desc('DB15K')
    save_rel2desc('DB15K')
