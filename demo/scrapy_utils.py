import requests


def get_dbpedia_summary(url):
    sparql_endpoint = 'http://dbpedia.org/sparql'
    sparql_query = """SELECT ?abs"""
