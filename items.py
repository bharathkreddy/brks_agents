import re
from transformers import AutoTokenizer


class Item:

    BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
    TOKENIZER = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    MIN_TOKENS = 150
    MAX_TOKENS = 160
    PROMPT = "How much the below product cost ? Give answer to the nearest dollar.\n\n"

    def __init__(self, data, price):
        self.price = price
        self.data = data

    @property
    def price(self):
        return self._price

    @price.setter
    def price(self, price):
        self._price = float(price)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        full_data = self.clean_data(data)        
        self._data = full_data


    def clean_data(self, data):
        title = data['title']
        description = " ".join(data['description'])
        features = " ".join(data['features'])
        full_data = title + description + features
        full_data = re.sub(r'\b\w*\d\w*\b', '',full_data)
        full_data = re.sub(r'\s+', ' ', full_data).strip()
        full_data = re.sub(r'([@;#-,!()/\<>.]),{2,}', r'\1',full_data)
        full_data = re.sub(r'([-–—]\s*){2,}', '', full_data)
        full_data = re.sub(r'[:\[\]"{}【】\s]+', ' ', full_data).strip()
        full_data = re.sub(r'([,]\s*){1,}', '', full_data)
        tokens = self.TOKENIZER.encode(self.PROMPT+full_data, add_special_tokens=False)
        full_data = self.TOKENIZER.decode(tokens[:self.MAX_TOKENS])
        full_data += '\n\nThe price is: $ '
        return full_data

    def prompt(self):
        return f'{self.data}{self.price}'
        


