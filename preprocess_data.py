import pandas as pd
from preprocessing import clean_comment
path = "/Users/falkne/PycharmProjects/storytelling/europolis_translations/"
english = path + "europolis_polish.csv"
german = path + "europolis_german.csv"
french = path + "europolis_french.csv"

en = pd.read_csv(english, sep="\t")
de = pd.read_csv(german, sep="\t")
fr = pd.read_csv(french, sep="\t")
print(en.columns)
print(de.columns)
print(fr.columns)


en = en[en.UniqueID != 1]
de = de[de.UniqueID != 1]
fr = fr[fr.UniqueID != 1]

en["orig_lang"] = len(en) * ["pl"]
de["orig_lang"] = len(de) * ["de"]
fr["orig_lang"] = len(fr) * ["fr"]
merged = pd.concat([en, de, fr])
print(list(merged.name))
english_comments = list(merged["translation"])
cleaned = []
for c in english_comments:
    clean_c = clean_comment(c)
    cleaned.append(clean_c)
merged["cleaned_comment"] = cleaned
print(len(merged))
merged.to_csv("europolis_dqi.csv", sep="\t")
"""
jlev: level of justification
(0) The speaker does not present any argument or only says that X should or should not be done, but no reason is given.
(1) Inferior Justification: Here a reason Y is given why X should or should not be done, but no linkage is made between X and Y—the inference is incomplete or the argument is merely supported with illustrations.
(2) Qualified Justification: A linkage is made why one should expect that X contributes to or detracts from Y. A single complete inference already qualifies for code 2.
(3)Sophisticated Justification (broad): At least two complete justifications are given, either two complete justifications for the same demand or complete justifications for two different demands.
(4) Sophisticated Justification (in depth): At least two complete justifications are given, either two complete justifications for the same demand or complete justifications for two different demands and discussed in d

int1: interactivity (refers to other ppls arguments)


jcon: reference to 'common good'

resp_gr: respect
"""