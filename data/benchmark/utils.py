import os
import json
import requests


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


def get_dbpedia_desc_json(name, lang='en'):
    url = f'https://dbpedia.org/data/{name}.json'
    response = requests.get(url)
    if response.status_code != 200:
        return None

    data = response.json()
    subject_uri = f'http://dbpedia.org/resource/{name}'

    abstracts = data.get(subject_uri, {}).get('http://dbpedia.org/ontology/abstract', [])
    for item in abstracts:
        if item.get('lang', {}) == lang:
            return item.get('value')
    return None


def save_ent2desc(data_dir):
    path = os.path.join(data_dir, 'entity.json')
    ent_num, ent2id, id2ent = load_entity(data_dir)
    res_dict = {}
    for ent in id2ent:
        ent_name = ent.split('/')[-1]
        desc = get_dbpedia_desc_json(ent_name, lang='en')
        ent_dict = {'name': ent_name, 'desc': desc}
        res_dict[ent] = ent_dict
    json.dump(res_dict, open(path, 'w', encoding='utf-8'))


def save_rel2desc(data_dir):
    path = os.path.join(data_dir, 'relation.json')
    rel_num, rel2id, id2rel = load_relation(data_dir)
    res_dict = {}
    for rel in id2rel:
        rel_name = rel.split('/')[-1]
        desc = get_dbpedia_desc_json(rel_name, lang='en')
        rel_dict = {'name': rel_name, 'desc': desc}
        res_dict[rel] = rel_dict
    json.dump(res_dict, open(path, 'w', encoding='utf-8'))


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
    # save_triple2text('DB15K', 'train2id.txt', 'train.txt')
    # save_triple2text('DB15K', 'test2id.txt', 'test.txt')
    # save_triple2text('DB15K', 'valid2id.txt', 'valid.txt')
    save_ent2desc('DB15K')
    pass
