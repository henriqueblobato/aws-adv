from pprint import pprint

ent = [{'BeginOffset': 12,
        'EndOffset': 25,
        'Score': 0.9662402868270874,
        'Text': 'outros países',
        'Type': 'QUANTITY'},
       {'BeginOffset': 320,
        'EndOffset': 326,
        'Score': 0.9802322387695312,
        'Text': 'Brasil',
        'Type': 'LOCATION'},
       {'BeginOffset': 424,
        'EndOffset': 432,
        'Score': 0.7151781320571899,
        'Text': 'momentos',
        'Type': 'QUANTITY'},
       {'BeginOffset': 532,
        'EndOffset': 541,
        'Score': 0.7145155072212219,
        'Text': 'eleitoral',
        'Type': 'DATE'},
       {'BeginOffset': 547,
        'EndOffset': 553,
        'Score': 0.7839928269386292,
        'Text': 'outros',
        'Type': 'QUANTITY'},
       {'BeginOffset': 574,
        'EndOffset': 588,
        'Score': 0.7589761018753052,
        'Text': 'mais tranquilo',
        'Type': 'QUANTITY'}]


def parse_entities(entities):
    response_dict = {}
    for entitie in entities:
        text = entitie.get('Text')
        type_name = entitie.get('Type')
        if type_name not in response_dict.keys():
            response_dict[type_name] = [{
                'Text': text,
                'Counter': 1,
            }]
        else:
            for i in response_dict[type_name]:
                if text not in i.values():
                    response_dict[type_name].append({
                        'Text': text,
                        'Counter': 1,
                    })
                    break
                else:
                    i['Counter'] += 1
                    break

    return response_dict
#
#
# p = parse_entities(ent)
# pprint(p)
