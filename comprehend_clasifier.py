import logging
import os
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pprint import pprint
from threading import Thread

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


def parse_entities(entities):
    response_dict = {}
    for entity in entities:
        text = entity.get('Text')
        key = entity.get('Type')
        score = entity.get('Score')
        score = round(score, 4)
        if key not in response_dict.keys():
            response_dict[key] = [{
                'text': text,
                'score': score,
                'count': 1
            }]
        else:
            values = [i.get('text') for i in response_dict[key]]
            if text not in values:
                response_dict[key].append({
                    'text': text,
                    'score': score,
                    'count': 1
                })
            else:
                for i in response_dict[key]:
                    if i.get('text') == text:
                        i['score'] = max(i['score'], score)
                        i['count'] += 1
    return response_dict


def get_top_key_phrases_by_score(key_phrases, top_n=3):
    top_key_phrases = []
    for key_phrase in key_phrases:
        score = key_phrase.get('Score')
        score = round(score, 5)
        top_key_phrases.append({
            'Text': key_phrase.get('Text'),
            'Score': score,
        })
    top_key_phrases.sort(key=lambda x: x['Score'], reverse=True)
    return top_key_phrases[:top_n]


class LanguageEnum(Enum):
    """
    The languages supported by Amazon Comprehend.
    """
    en = 'en'
    es = 'es'
    fr = 'fr'
    de = 'de'
    it = 'it'
    pt = 'pt'
    ja = 'ja'
    zh = 'zh'


class ComprehendDetect:
    """Encapsulates Comprehend detection functions."""

    def __init__(self):
        """
        :param comprehend_client: A Boto3 Comprehend client.
        """
        self.comprehend_client = boto3.client(
            'comprehend',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION')
        )

    def detect_languages(self, text):
        """
        Detects languages used in a document.

        param text: The document to inspect.
        :return: The list of languages along with their confidence scores.
        """
        try:
            response = self.comprehend_client.detect_dominant_language(Text=text)
            languages = response['Languages']
            logger.info("Detected %s languages.", len(languages))
        except ClientError:
            logger.exception("Couldn't detect languages.")
            raise
        else:
            return languages

    def detect_pii(self, text, language_code, results, demo_size):
        """
        Detects PII in a document. PII is personal information that can be
        identified from the text.

        :param text: The document to inspect.
        :param language_code: The language of the document.
        :return: The list of entities along with their confidence scores.
        """
        # if not language_code == LanguageEnum.en.value:
        #     logger.warning("PII detection only works for English.")
        #     return {'PII': []}
        try:
            response = self.comprehend_client.detect_pii_entities(
                Text=text,
                LanguageCode='en'
            )
            piis = response['Entities']
            return_response = []
            for entity in piis:
                score = entity.get('Score')
                begin_offset = entity.get('BeginOffset')
                end_offset = entity.get('EndOffset')
                entity_text = text[begin_offset:end_offset]
                entity_type = entity.get('Type')
                return_response.append({
                    'Text': entity_text,
                    'Score': score,
                    'Type': entity_type
                })
            piis = parse_entities(return_response)
            logger.info("Detected %s entities.", len(piis))
        except ClientError:
            logger.exception("Couldn't detect entities.")
            raise
        else:
            results.append({'PII': piis})
            return piis

    def detect_entities(self, text, language_code, results, demo_size):
        """
        Detects entities in a document. Entities can be things like people and places
        or other common terms.

        :param text: The document to inspect.
        :param language_code: The language of the document.
        :return: The list of entities along with their confidence scores.
        """
        try:
            response = self.comprehend_client.detect_entities(
                Text=text, LanguageCode=language_code)
            entities = response['Entities']
            entities = parse_entities(entities)
            logger.info("Detected %s entities.", len(entities))
        except ClientError:
            logger.exception("Couldn't detect entities.")
            raise
        else:
            results.append({'Entities': entities})
            return entities

    def detect_key_phrases(self, text, language_code, results, demo_size):
        """
        Detects key phrases in a document. A key phrase is typically a noun and its
        modifiers.

        param text: The document to inspect.
        param language_code: The language of the document.
        :return: The list of key phrases along with their confidence scores.
        """
        try:
            response = self.comprehend_client.detect_key_phrases(
                Text=text, LanguageCode=language_code)
            phrases = response['KeyPhrases']
            phrases = get_top_key_phrases_by_score(phrases, top_n=30)
            logger.info("Detected %s phrases.", len(phrases))
        except ClientError:
            logger.exception("Couldn't detect phrases.")
            raise
        else:
            results.append({'KeyPhrases': phrases})
            return phrases

    def detect_sentiment(self, text, language_code, results, demo_size):
        """
        Detects the overall sentiment expressed in a document. Sentiment can
        be positive, negative, neutral, or a mixture.

        param text: The document to inspect.
        param language_code: The language of the document.
        :return: The sentiments along with their confidence scores.
        """
        try:
            response = self.comprehend_client.detect_sentiment(
                Text=text,
                LanguageCode=language_code
            )
            response = {
                a: round(b, 4)
                for a, b in response.get('SentimentScore').items()
                if isinstance(b, float)
            }

            logger.info("Detected primary sentiment %s.", response)
        except ClientError:
            logger.exception("Couldn't detect sentiment.")
            raise
        else:
            results.append({'Sentiment': response})
            return response

    def detect_syntax(self, text, language_code, results, demo_size):
        """
        Detects syntactical elements of a document. Syntax tokens are portions of
        text along with their use as parts of speech, such as nouns, verbs, and
        interjections.

        param text: The document to inspect.
        param language_code: The language of the document.
        :return: The list of syntax tokens along with their confidence scores.
        """
        try:
            response = self.comprehend_client.detect_syntax(
                Text=text, LanguageCode=language_code)
            tokens = response['SyntaxTokens']
            logger.info("Detected %s syntax tokens.", len(tokens))
        except ClientError:
            logger.exception("Couldn't detect syntax.")
            raise
        else:
            results.append({'Syntax': tokens[:demo_size]})
            return tokens

    def detect(self, threads=1, demo_size=10, language_code=LanguageEnum.en):
        """
        Detects entities, key phrases, sentiments, and syntax in a document.

        param threads: The number of threads to use.
        :return: A list of dictionaries containing the results of the analysis.
        """
        results = []
        with ThreadPoolExecutor(max_workers=threads) as executor:
            for text in self.texts:
                if language_code:
                    language_code = language_code[0]['LanguageCode']
                    entities = executor.submit(
                        self.detect_entities, text, language_code, results, demo_size)
                    key_phrases = executor.submit(
                        self.detect_key_phrases, text, language_code, results, demo_size)
                    sentiment = executor.submit(
                        self.detect_sentiment, text, language_code, results, demo_size)
                    syntax = executor.submit(
                        self.detect_syntax, text, language_code, results, demo_size)
                    entities.result()
                    key_phrases.result()
                    sentiment.result()
                    syntax.result()
        return results


def usage_demo():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    comp_detect = ComprehendDetect()

    with open('detect_sample.txt') as sample_file:
        sample_text = sample_file.read()
        sample_text = sample_text.replace('\n', ' ')

    demo_size = 3

    languages = comp_detect.detect_languages(sample_text)
    lang_code = languages[0]['LanguageCode']

    functions = [
        comp_detect.detect_entities,
        comp_detect.detect_key_phrases,
        comp_detect.detect_sentiment,
        comp_detect.detect_syntax,
        comp_detect.detect_pii,
    ]

    results = []
    for i in functions:
        language = getattr(LanguageEnum, lang_code).value
        thread = Thread(
            target=i,
            args=(sample_text, language, results, demo_size)
        )
        thread.start()
        thread.join()

    pprint(results)


# if __name__ == '__main__':
#     usage_demo()
