from sarvamai import SarvamAI
import os

class SarvamClient:
    def __init__(self):
        self.client = SarvamAI(
            api_subscription_key=os.getenv("SARVAMAI_API_KEY")
            )

    def speech_to_text(self, audio_file_path):
        '''
        Function needs to be expanded as per need
        '''
        # This will work for small audio files.
        response = self.client.speech_to_text.translate(
            file=open(audio_file_path, "rb"),
            model="saaras:v2"
            )

        # for larger files, we need to implement chunking
        # There is a reference here: https://docs.sarvam.ai/api-reference-docs/cookbook/starter-notebooks/stt-translate-api-tutorial#4-speech-to-text-translation-api
        return response

    def text_to_speech(self, text, voice):
        '''
        This function also needs to be expanded as per need
        https://docs.sarvam.ai/api-reference-docs/cookbook/starter-notebooks/tts-api-tutorial
        '''
        response = client.text_to_speech.convert(
            text="Your Text",
            target_language_code="hi-IN",
            speaker="anushka",
            enable_preprocessing=True
        )
        # play(response)
        save(response, "output.wav")
        return response