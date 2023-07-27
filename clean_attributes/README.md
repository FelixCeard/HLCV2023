# The attributes extracted by LENS are not very nice :(

We have to prune many attributes, cause:
1. each image has 20 attributes. Why 20? why not more, or less?
2. some attributes describe the same thing (only very slight variation)
3. Many attributes do not make sense (e.g. `3x3`, `Audi rs 6`)

## Methodology
1. Check whether the word is present in the dictionary
    - Maybe very concervative (small derivations will be missed)
2. Check whether it conatins a word that is present in the english dictionary
   - might extract only adjectives and miss the big picture
3. Named-entity recognition
   - should give the best result but very expensive

