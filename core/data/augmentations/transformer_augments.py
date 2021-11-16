from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class AugTransformer():
    
    def __init__(self, allow_backtranslate=True):
        self.generator = pipeline('text-generation', model='gpt2')
        
        self.allow_backtranslate = allow_backtranslate
        if self.allow_backtranslate:
            # English to German using the Pipeline and T5
            self.translator_en_to_de = pipeline("translation_en_to_de", model='t5-base')

            # German to English using Bert2Bert model
            self.tokenizer_de_to_en = AutoTokenizer.from_pretrained("google/bert2bert_L-24_wmt_de_en", pad_token="<pad>", eos_token="</s>", bos_token="<s>")
            self.model_de_to_en = AutoModelForSeq2SeqLM.from_pretrained("google/bert2bert_L-24_wmt_de_en")
        
    def backtranslate(self, sentence):
        augmented_text = sentence
        if self.allow_backtranslate:
            en_to_de_output = self.translator_en_to_de(sentence)
            translated_text = en_to_de_output[0]['translation_text']

            input_ids = self.tokenizer_de_to_en(translated_text, return_tensors="pt", add_special_tokens=False).input_ids
            output_ids = self.model_de_to_en.generate(input_ids)[0]
            augmented_text = self.tokenizer_de_to_en.decode(output_ids, skip_special_tokens=True)
        else:
            print("Backtranslate was not allowed")
        
        return augmented_text
    
    def generate(self, sentence, num_new_words):
        input_length = len(sentence.split())
        output_length = input_length + num_new_words
        gpt_output = self.generator(sentence, max_length=output_length, num_return_sequences=5)
        augmented_text = gpt_output[0]['generated_text']
        
        return augmented_text