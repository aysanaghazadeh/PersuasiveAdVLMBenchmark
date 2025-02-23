atypical_id_to_relation_name = {
    "1": "TR1",
    "2": "TR2",
    "3": "OIO",
    "8": "OR",
}

atypical_relation_name_to_id = {
    "TR1": "1",
    "TR2": "2",
    "OIO": "3",
    "OR": "8",
}

atypicality_def = {
    'OIO': '{primary_concept} is visibly located within {secondary_concept}, in an unconventional manner.',
    'TR1': 'The surface of {primary_concept} mimics the texture of {secondary_concept}, while retaining its original structure.',
    'TR2': '{primary_concept} appears to be composed of numerous, smaller instances of {secondary_concept}, altering its texture.',
    'OR': '{primary_concept} completely replaces {secondary_concept} in its usual context, assuming its function or position.'
}

atypicality_def_1 = """
        1) Object Inside Object (OIO): An object completely is inside of another object. e.g. auto racing inside of a car.
        2) Texture Replacement 1 (TR1): Objects’ texture borrowed from another objects. e.g., apple texture made of kiwi.
        3) Texture Replacement 2 (TR2): Texture created by combining several small objects. e.g. Beer glass is made of multiple tiny ice cubes.
        4) Object Replacement (OR): An object is depicted in a role or context typically filled by a different object. e.g. cigarettes placed in the a gun where bullets occur.
        """
atypicality_def_2 = """
        1) Object Inside Object (OIO): When one thing is completely  inside another thing where you wouldn't expect it, like a race car inside a another regular car.
        2) Texture Replacement 1 (TR1): When the outside of something looks like it's made from a completely different thing, like an apple with the skin of a kiwi.
        3) Texture Replacement 2 (TR2): When something is made from lots of small things that are not usually part of it, like a beer glass made of tiny ice cubes.
        4) Object Replacement (OR): When one thing is used in a place or way where you usually find another thing, like cigarettes in a gun where bullets normally are.
        """


atypicality_def_3 = """
        1) Object Inside Object (OIO): When one thing is completely inside another thing where it is not common or natural.
        2) Texture Replacement 1 (TR1): When the skin/texture of an object is replaced with another object to inherit an attribute of that.
        3) Texture Replacement 2 (TR2): When something is made from lots of small things that are not usually part of it to inherit an attribute of small objects.
        4) Object Replacement (OR): When one thing is used in a place or way where you usually find another thing to act as the original object.
        """


relation_stmt_1 = {
    'OIO': ' is partly or totally in the ',
    'TR1': "'s texture is borrowed from single ",
    'TR2': "'s texture is made of multiple tiny ",
    'OR': " is in the place of "
}

relation_stmt_2 = {
    'OIO': ' placed in ',
    'TR1': " is shown by single ",
    'TR2': " is shown by multiple tiny ",
    'OR': " has replaced "

}

relation_stmt_3 = {
    'OIO': " is in the ",
    'TR1': "'s texture  is made of the texture of",
    'TR2': "'s texture is made of multiple tiny ",
    'OR': " is shown in the place of "
}

relation_def_4 = {
    'OIO': 'One object is in another object which is impossible in the real world',
    'TR1': 'One object is made of the texture of another object which does not happen in real world',
    'TR2': "One object's texture is made of multiple tiny instance from another object",
    'OR': "One object replaces another object"
}

relation_def_5 = {
    'OIO': 'One object is in another object which is impossible in the real world',
    'TR1': 'Texture of one object is made of the texture of another object which does not happen in real world',
    'TR2': "One object's texture is made of multiple tiny instance from another object",
    'OR': "One object is in the place another object"
}

relation_def_6 = {
    'OIO': 'Part of or the whole of one of the object is in another object',
    'TR1': "One object's texture is borrowed from another object",
    'TR2': "One object is made of many tiny objects",
    'OR': "One object is shown in the place of another object"
}
