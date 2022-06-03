from pprint import pprint

import boto3

from e1 import parse_entities

LANGUAGE = 'pt'


def comprehend_client():
    return boto3.client(service_name='comprehend', region_name='us-east-2')


client = comprehend_client()


def detect_entities(text):
    return comprehend_client().detect_entities(Text=text, LanguageCode=LANGUAGE)


def detect_key_phrases(text):
    return comprehend_client().detect_key_phrases(Text=text, LanguageCode=LANGUAGE)


def detect_sentiment(text):
    res = comprehend_client().detect_sentiment(Text=text, LanguageCode=LANGUAGE)
    res.pop('ResponseMetadata')
    return res


def detect_syntax(text):
    return comprehend_client().detect_syntax(Text=text, LanguageCode=LANGUAGE)


def text_analysis(text):
    entities = detect_entities(text).get('Entities')
    entities = parse_entities(entities)
    all = {
        'entities': entities,
        'sentiment': detect_sentiment(text),
    }

    return all


response = text_analysis('''Investir em outros países diversifica as aplicações e faz com que o risco fique "diluído" 
em relação à variação de preços provocadas por políticas locais. Isso vale tanto para investimentos em mercados 
desenvolvidos quanto em desenvolvimento. Mirar nos emergentes é uma forma de reduzir a concentração da carteira no 
Brasil e, ao mesmo tempo, investir em ativos com potencial de lucro. Com aplicações em países que vivem momentos 
diferentes não apenas na economia, mas também na política (combinando, por exemplo, locais em ano eleitoral com 
outros em momento político mais tranquilo), a ideia é atenuar o sobe e desce das ações.''')
pprint(response)
