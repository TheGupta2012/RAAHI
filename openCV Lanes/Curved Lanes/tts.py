import pyttsx3

def speak(text, rate=1.4,voice_type="female",volume=1):
    engine=pyttsx3.init()   
    # speech rate
    engine.setProperty('rate',rate*engine.getProperty('rate'))

    # Voice
    if(voice_type=="female"):
        engine.setProperty('voice',engine.getProperty('voices')[1].id)
    else:
        engine.setProperty('voice',engine.getProperty('voices')[0].id)

    # Volume)   
    engine.setProperty('volume',volume) 
    engine.say(text)
    engine.runAndWait()


