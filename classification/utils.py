def remove_emphasis(word):
    upper_vowels = 'ΑΕΟΩΥΙΗ'
    lower_vowels = 'αεοωυιη'
    emphasized_upper_vowels = "ΆΈΌΏΎΊΉ"
    emphasized_lower_vowels = 'άέόώύίή'
    no_dialitika_vowels_upper = 'ΙΥ'
    dialitika_vowels_upper = 'ΪΫ'
    dialitika_lower_vowels = 'ϊϋ'
    emphasized_dialitika_lower_vowels = 'ΐΰ'

    capital_candidate = word[0]
    # check if emphasized upper
    emphasized_upper_idx = emphasized_upper_vowels.find (capital_candidate)
    if emphasized_upper_idx != -1:
        word = word.replace(capital_candidate,upper_vowels[emphasized_upper_idx])

    # check if dialitika emphasized upper
    dialitika_emphasized_upper_idx = dialitika_vowels_upper.find (capital_candidate)
    if dialitika_emphasized_upper_idx != -1:
        word = word.replace(capital_candidate,no_dialitika_vowels_upper [dialitika_emphasized_upper_idx])

    # check if emphasized upper
    for idx, char in enumerate(word):
        emphasized_lower_idx = emphasized_lower_vowels.find (char)
        if emphasized_lower_idx != -1:
            word = word.replace(char, lower_vowels[emphasized_lower_idx])
            break

    # check if dialitika lower
    for idx, char in enumerate(word):
        dialitika_lower_idx = dialitika_lower_vowels.find (char)
        if dialitika_lower_idx != -1:
            word = word.replace(char,no_dialitika_vowels_upper[dialitika_lower_idx])
        emphasized_dialitika_lower_idx = emphasized_dialitika_lower_vowels.find (char)
        if emphasized_dialitika_lower_idx != -1:
            word = word.replace(char,no_dialitika_vowels_upper [emphasized_dialitika_lower_idx])
    return word
