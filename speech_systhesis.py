from gtts import gTTS
import os

text="i love python and ml"
lang='en'

out=gTTS(text=text,lang=lang)

out.save('out.mp3')
os.system('start out.mp3')
print('removing mp3 file')
os.remove('out.mp3')
