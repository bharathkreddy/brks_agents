from typing import Optional
from transformers import AutoTokenizer
import re

BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"

MIN_TOKENS = 150 # Any less than this, and we don't have enough useful content
MAX_TOKENS = 160 # Truncate after this many tokens. Then after adding in prompt text, we will get to around 180 tokens

MIN_CHARS = 300
CEILING_CHARS = MAX_TOKENS * 7

class Item:
    """
    An Item is a cleaned, curated datapoint of a Product with a Price
    """
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    PREFIX = "Price is $"
    QUESTION = "How much does this cost to the nearest dollar?"
    REMOVALS = ['"Batteries Included?": "No"', '"Batteries Included?": "Yes"', '"Batteries Required?": "No"', '"Batteries Required?": "Yes"', "By Manufacturer", "Item", "Date First", "Package", ":", "Number of", "Best Sellers", "Number", "Product "]

    title: str
    price: float
    category: str
    token_count: int = 0
    details: Optional[str]
    prompt: Optional[str] = None
    include = False

    def __init__(self, data, price):
        self.title = data['title']
        self.price = price
        self.parse(data)

    def scrub_details(self):
        """
        Clean up the details string by removing common text that doesn't add value
        """
        details = self.details
        for remove in self.REMOVALS:
            details = details.replace(remove, "")
        return details

    def scrub(self, stuff):
        """
        Clean up the provided text by removing unnecessary characters and whitespace
        Also remove words that are 7+ chars and contain numbers, as these are likely irrelevant product numbers
        """
        stuff = re.sub(r'[:\[\]"{}【】\s]+', ' ', stuff).strip()
        stuff = stuff.replace(" ,", ",").replace(",,,",",").replace(",,",",")
        words = stuff.split(' ')
        select = [word for word in words if len(word)<7 or not any(char.isdigit() for char in word)]
        return " ".join(select)
    
    def parse(self, data):
        """
        Parse this datapoint and if it fits within the allowed Token range,
        then set include to True
        """
        contents = '\n'.join(data['description'])
        if contents:
            contents += '\n'
        features = '\n'.join(data['features'])
        if features:
            contents += features + '\n'
        self.details = data['details']
        if self.details:
            contents += self.scrub_details() + '\n'
        if len(contents) > MIN_CHARS:
            contents = contents[:CEILING_CHARS]
            text = f"{self.scrub(self.title)}\n{self.scrub(contents)}"
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) > MIN_TOKENS:
                tokens = tokens[:MAX_TOKENS]
                text = self.tokenizer.decode(tokens)
                self.make_prompt(text)
                self.include = True

    def make_prompt(self, text):
        """
        Set the prompt instance variable to be a prompt appropriate for training
        """
        self.prompt = f"{self.QUESTION}\n\n{text}\n\n"
        self.prompt += f"{self.PREFIX}{str(round(self.price))}.00"
        self.token_count = len(self.tokenizer.encode(self.prompt, add_special_tokens=False))

    def test_prompt(self):
        """
        Return a prompt suitable for testing, with the actual price removed
        """
        return self.prompt.split(self.PREFIX)[0] + self.PREFIX

    def __repr__(self):
        """
        Return a String version of this Item
        """
        return f"<{self.title} = ${self.price}>"

        

    
    

# class Item:

#     BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
#     TOKENIZER = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
#     MIN_TOKENS = 150
#     MAX_TOKENS = 160
#     PROMPT = "How much the below product cost ? Give answer to the nearest dollar.\n\n"

#     def __init__(self, data, price):
#         self.price = price
#         self.data = data

#     @property
#     def price(self):
#         return self._price

#     @price.setter
#     def price(self, price):
#         self._price = float(price)

#     @property
#     def data(self):
#         return self._data

#     @data.setter
#     def data(self, data):
#         full_data = self.clean_data(data)        
#         self._data = full_data


#     def clean_data(self, data):
#         title = data['title']
#         description = " ".join(data['description'])
#         features = " ".join(data['features'])
#         full_data = title + description + features
#         full_data = re.sub(r'\b\w*\d\w*\b', '',full_data)
#         full_data = re.sub(r'\s+', ' ', full_data).strip()
#         full_data = re.sub(r'([@;#-,!()/\<>.]),{2,}', r'\1',full_data)
#         full_data = re.sub(r'([-–—]\s*){2,}', '', full_data)
#         full_data = re.sub(r'[:\[\]"{}【】\s]+', ' ', full_data).strip()
#         full_data = re.sub(r'([,]\s*){1,}', '', full_data)
#         tokens = self.TOKENIZER.encode(self.PROMPT+full_data, add_special_tokens=False)
#         full_data = self.TOKENIZER.decode(tokens[:self.MAX_TOKENS])
#         full_data += '\n\nThe price is: $ '
#         return full_data

#     def prompt(self):
#         return f'{self.data}{self.price}'
        


