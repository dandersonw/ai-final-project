def transform_name(datum):
    name = datum[1]
    returnVal = []
    for c in name:
        returnVal.append(ord(c))
    return returnVal

# Entity = 1, Spell = 2, Enchantment = 3, Artifact = 4
label_mapping = {
    "Creature": 1,
    "Planeswalker": 1,
    "Instant": 2,
    "Sorcery": 2,
    "Enchantment": 3,
    "Artifact": 4
}

def transform_data(datum):
    record = {}
    record['token'] = transform_name(datum)
    record['length'] = len(datum[1])
    record['image'] = datum[4]
    record['label'] = label_mapping[datum[6]]
    return record
