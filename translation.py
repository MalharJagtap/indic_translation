import tkinter as tk
from tkinter import ttk
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor
import pygame
import os
import tempfile
import threading
import soundfile as sf
from parler_tts import ParlerTTSForConditionalGeneration


class TranslationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("IndicTransToolkit Translator with TTS")

        # Initialize pygame for audio playback
        pygame.mixer.init()

        # Audio playback state
        self.playing_audio = False
        self.temp_audio_file = None
        self.tts_thread = None

        # Update model and tokenizer setup to use the new model
        self.model_name = "ai4bharat/indictrans2-en-indic-dist-200M"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, trust_remote_code=True)
        # Determine device and move model to it. This is the fix.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        print(f"Using device: {self.device}")

        self.ip = IndicProcessor(inference=True)

        # TTS model initialization (will be loaded on demand)
        self.tts_model = None
        self.tts_tokenizer = None
        self.description_tokenizer = None
        self.tts_loaded = False

        # GUI Elements
        self.create_widgets()

        # Status bar for loading information
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_widgets(self):
        # Input text
        self.input_label = ttk.Label(self.root, text="Input Text:")
        self.input_label.pack(pady=5)
        self.input_text = tk.Text(self.root, height=10, width=50)
        self.input_text.pack(pady=5)

        # Supported languages
        languages = [
            "eng_Latn", "hin_Deva", "ben_Beng", "guj_Gujr", "kan_Knda",
            "mal_Mlym", "mar_Deva", "pan_Guru", "tam_Taml", "tel_Telu",
            "asm_Beng", "bod_Tibt", "doi_Deva", "gom_Deva", "kas_Arab",
            "kas_Deva", "mai_Deva", "mni_Beng", "mni_Mtei", "nep_Deva",
            "san_Deva", "sat_Beng", "sat_Olck", "urd_Arab"
        ]

        # Source and target language selection
        self.src_lang_label = ttk.Label(self.root, text="Source Language:")
        self.src_lang_label.pack(pady=5)
        self.src_lang = ttk.Combobox(self.root, values=languages, state="readonly")
        self.src_lang.pack(pady=5)
        self.src_lang.set("eng_Latn")

        self.tgt_lang_label = ttk.Label(self.root, text="Target Language:")
        self.tgt_lang_label.pack(pady=5)
        self.tgt_lang = ttk.Combobox(self.root, values=languages, state="readonly")
        self.tgt_lang.pack(pady=5)
        self.tgt_lang.set("hin_Deva")

        # Translate button
        self.translate_button = ttk.Button(self.root, text="Translate", command=self.translate_text)
        self.translate_button.pack(pady=10)

        # Output text
        self.output_label = ttk.Label(self.root, text="Translated Text:")
        self.output_label.pack(pady=5)
        self.output_text = tk.Text(self.root, height=10, width=50)
        self.output_text.pack(pady=5)

        # TTS controls frame
        self.tts_frame = ttk.Frame(self.root)
        self.tts_frame.pack(pady=10)

        # TTS buttons
        self.play_button = ttk.Button(self.tts_frame, text="Play Audio (TTS)", command=self.play_audio)
        self.play_button.pack(side=tk.LEFT, padx=5)
        self.play_button.config(state=tk.DISABLED)

        self.stop_button = ttk.Button(self.tts_frame, text="Stop Audio", command=self.stop_audio)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        self.stop_button.config(state=tk.DISABLED)

    def load_tts_model(self):
        """Load the TTS model on demand"""
        if not self.tts_loaded:
            try:
                self.status_var.set("Loading TTS model... This may take a moment.")
                self.root.update()

                # Load the locally downloaded TTS model
                tts_model_path = "C:\\Users\\Malhar\\PycharmProjects\\indic\\indic-parler-tts-pretrained"
                self.tts_model = ParlerTTSForConditionalGeneration.from_pretrained(tts_model_path)
                self.tts_tokenizer = AutoTokenizer.from_pretrained(tts_model_path)
                self.description_tokenizer = AutoTokenizer.from_pretrained(
                    self.tts_model.config.text_encoder._name_or_path)

                # Move TTS model to appropriate device
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.tts_model = self.tts_model.to(device)

                # (Optional) If needed, ensure the translation model is on device as well.
                self.device = torch.device(device)
                self.model = self.model.to(self.device)

                self.tts_loaded = True
                self.status_var.set("TTS model loaded successfully.")
                return True
            except Exception as e:
                self.status_var.set(f"Error loading TTS model: {str(e)}")
                print(f"Detailed error: {e}")
                import traceback
                traceback.print_exc()
                return False
        return True

    def translate_text(self):
        input_sentences = [self.input_text.get("1.0", tk.END).strip()]
        src_lang = self.src_lang.get()
        tgt_lang = self.tgt_lang.get()

        # Preprocess the input sentences
        batch = self.ip.preprocess_batch(input_sentences, src_lang=src_lang, tgt_lang=tgt_lang)

        # Determine the device to use (should match the device set in __init__)
        DEVICE = self.device

        # Tokenize the sentences and generate input encodings
        inputs = self.tokenizer(batch, truncation=True, padding="longest", return_tensors="pt",
                                return_attention_mask=True).to(DEVICE)

        # Generate translations using the model
        with torch.no_grad():
            generated_tokens = self.model.generate(**inputs, use_cache=True, min_length=0, max_length=256, num_beams=5,
                                                   num_return_sequences=1)

        # Decode the generated tokens into text
        with self.tokenizer.as_target_tokenizer():
            generated_tokens = self.tokenizer.batch_decode(generated_tokens.detach().cpu().tolist(),
                                                           skip_special_tokens=True, clean_up_tokenization_spaces=True)

        # Postprocess the translations, including entity replacement
        translations = self.ip.postprocess_batch(generated_tokens, lang=tgt_lang)

        # Display the translated text
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, translations[0])

        # Enable play button if translation is available
        if translations[0].strip():
            self.play_button.config(state=tk.NORMAL)
        else:
            self.play_button.config(state=tk.DISABLED)

    def generate_audio_in_thread(self, text, lang_code):
        try:
            # Load TTS model if not already loaded
            if not self.load_tts_model():
                self.root.after(0, self.reset_audio_buttons)
                return

            # Create a temporary file for the audio
            fd, self.temp_audio_file = tempfile.mkstemp(suffix='.wav')
            os.close(fd)

            # Map language code to speaker and description
            speaker, description = self.get_speaker_and_description(lang_code)

            # Prepare inputs for the TTS model
            device = next(self.tts_model.parameters()).device

            # Tokenize the description and prompt
            description_input_ids = self.description_tokenizer(description, return_tensors="pt").to(device)
            prompt_input_ids = self.tts_tokenizer(text, return_tensors="pt").to(device)

            # Generate audio
            with torch.no_grad():
                generation = self.tts_model.generate(
                    input_ids=description_input_ids.input_ids,
                    attention_mask=description_input_ids.attention_mask,
                    prompt_input_ids=prompt_input_ids.input_ids,
                    prompt_attention_mask=prompt_input_ids.attention_mask
                )

            # Convert to numpy and save
            audio_arr = generation.cpu().numpy().squeeze()
            sf.write(self.temp_audio_file, audio_arr, self.tts_model.config.sampling_rate)

            # Play the audio on the main thread
            self.root.after(0, self.play_generated_audio)

        except Exception as e:
            print(f"Error generating audio: {e}")
            import traceback
            traceback.print_exc()
            self.status_var.set(f"Error generating audio: {str(e)}")
            self.root.after(0, self.reset_audio_buttons)

    def get_speaker_and_description(self, lang_code):
        """Get appropriate speaker and description for the given language"""
        # Map language codes to recommended speakers and create appropriate descriptions
        lang_to_speaker = {
            "asm_Beng": "Amit",
            "ben_Beng": "Arjun",
            "bod_Tibt": "Bikram",
            "eng_Latn": "Thoma",
            "guj_Gujr": "Yash",
            "hin_Deva": "Rohit",
            "kan_Knda": "Suresh",
            "mal_Mlym": "Anjali",
            "mar_Deva": "Sanjay",
            "nep_Deva": "Amrita",
            "pan_Guru": "Divjot",
            "san_Deva": "Aryan",
            "tam_Taml": "Jaya",
            "tel_Telu": "Prakash",
            "urd_Arab": "Rohit"  # Fallback to Hindi speaker for Urdu
        }

        # Get speaker for the language or default to English speaker
        speaker = lang_to_speaker.get(lang_code, "Thoma")

        # Create a description with the speaker
        description = f"{speaker}'s voice is clear and expressive with a moderate speed and pitch. The recording is of very high quality with no background noise."

        return speaker, description

    def play_generated_audio(self):
        if os.path.exists(self.temp_audio_file):
            try:
                pygame.mixer.music.load(self.temp_audio_file)
                pygame.mixer.music.play()

                # Update button states
                self.playing_audio = True
                self.play_button.config(state=tk.DISABLED)
                self.stop_button.config(state=tk.NORMAL)

                # Set up a callback for when audio finishes playing
                self.root.after(100, self.check_audio_status)

                self.status_var.set("Playing audio...")
            except Exception as e:
                print(f"Error playing audio: {e}")
                self.status_var.set(f"Error playing audio: {str(e)}")
                self.reset_audio_buttons()
        else:
            self.status_var.set("Audio file not found.")
            self.reset_audio_buttons()

    def play_audio(self):
        # Get the translated text and target language
        text = self.output_text.get("1.0", tk.END).strip()
        tgt_lang = self.tgt_lang.get()

        if not text:
            self.status_var.set("No text to synthesize.")
            return

        # Stop any currently playing audio
        self.stop_audio()

        # Disable play button while generating audio
        self.play_button.config(state=tk.DISABLED)
        self.status_var.set("Generating audio...")

        # Generate audio in a separate thread to avoid freezing the UI
        self.tts_thread = threading.Thread(
            target=self.generate_audio_in_thread,
            args=(text, tgt_lang)
        )
        self.tts_thread.daemon = True
        self.tts_thread.start()

    def check_audio_status(self):
        if self.playing_audio and not pygame.mixer.music.get_busy():
            # Audio finished playing
            self.status_var.set("Audio playback completed.")
            self.reset_audio_buttons()
        elif self.playing_audio:
            # Audio still playing, check again later
            self.root.after(100, self.check_audio_status)

    def reset_audio_buttons(self):
        self.playing_audio = False
        self.play_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.cleanup_temp_file()

    def stop_audio(self):
        if self.playing_audio:
            pygame.mixer.music.stop()
            self.status_var.set("Audio playback stopped.")
            self.reset_audio_buttons()

    def cleanup_temp_file(self):
        if self.temp_audio_file and os.path.exists(self.temp_audio_file):
            try:
                os.remove(self.temp_audio_file)
                self.temp_audio_file = None
            except Exception as e:
                print(f"Error removing temporary file: {e}")

    def __del__(self):
        # Clean up resources when the app is closed
        self.cleanup_temp_file()
        pygame.mixer.quit()


if __name__ == "__main__":
    root = tk.Tk()
    app = TranslationApp(root)
    root.mainloop()