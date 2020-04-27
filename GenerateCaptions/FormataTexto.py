import unicodedata
import re

"""
A remoção de acentos foi baseada em uma resposta no Stack Overflow.
http://stackoverflow.com/a/517974/3464573
"""

f = open("merged-file","r") 
ptTxt = f.read()
ptTxt = ptTxt.splitlines()
f.close()

f = open("pt-captions.txt","w") 

for caption in ptTxt:
    # Unicode normalize transforma um caracter em seu equivalente em latin.
    nfkd = unicodedata.normalize('NFKD', caption)
    palavraSemAcento = u"".join([c for c in nfkd if not unicodedata.combining(c)])

    # Usa expressão regular para retornar a palavra apenas com números, letras e espaço
    f.write(re.sub('[^a-zA-Z0-9 \\\]', '', palavraSemAcento).lower()+"\n")

f.close()