# EXPERIMENTS WITH AGENTIC AI

### Ways to use models

1. Chat interfaces (openAI, perplexity etc)
2. Cloud APIs
   - LLM API (openai's api, claude etc)
   - Managed AI cloud services (Amazon Bedrock, Google Vertex AI, Azure ML, groq etc.)
3. Direct inference
   - With hugging face transformers library - weights directly used by frameworks like pytorch etc.
   - Ollama to run locally.

NOTE: To replicate this work -

1. git clone the repo
2. run `uv sync` (if you dont use UV - perhaps you should check it out.)
3. Now you would have same env, vars, python version and libraries as i have.

### [Single agent experiments](text_summarization_beautifulSoup.ipynb)

1. Explores usage of a few tools to scrape websites beyond just `requests` library

1. `selenium` : plain vanila scrapping but user control on entire website.
1. `chromium web driver` : more control when page is rendered via js frameworks.
1. `undetected chromium` : by-pass captha, can be used to bypass ip-blocking, geolocks etc if enhanced.

1. Scraped site is parsed using Beautiful Soup
1. Parsed content is passed to an llm for summarization.
1. `openAI API` : can be swapped with any other provider.
1. `ollama` : locally running models on ollama.
1. A simple code co-pilot with streaming responses.
1. Adversarial chat between `openAI` and `claude` to improve a tic-tac game code.

### [multi agent experiments](mulitmodel_agents.ipynb)

Building a multi-model chatbot with Streaming AI ui, memory managed, context enhanced, tool enabled, multi shot prompt enhanced, conversational features.

1. AI chat interface with Gradio, with streaming
2. Memory management
3. Multi-shot prompting and context enrichment
4. Providing tools, building custom tools, allowing code execution by models.
5. Integrating Image and Sound generation.

### Transformers and everything about them

1. [Huggingface pipelines](Trasnformers.ipynb) (see hugging face transformers repo to understand default models for each task.). I could use pipelines for all types of things where i did not need much control over underlying internals. Good for most of tasks, just identify the task, a model good for it and huggingface transformers library abstracts everything away !. I tried all default models for
   - Sentiment analysis
   - ner
   - classifier
   - Q & A
   - Summarizer
   - translation
   - zero-shot-classifier
   - text generation
   - image generation
   - text-to-speech models.
2. `tokenizers` : When Deeper control is required. I learnt these

   - Create tokenizers for different models.
     - vocab, added cocab of tokenizers, encoding and decoding.
     - comparing tokenizers for `instruct` variant of llama-3b, phi-3 and Qwen models. `instruct` variants have been trained to expect `chat-prompts`.
     - See tokenizer for code only models like `starcoder2-15b`
   - Translate between text and tokens
   - Understand special tokens and chat templates

3. ## Loading & quantization of models with bitsandbytes.

   I did these experiments on a colab t4 instance and it was fairly fast. (~ 5 odd mins for quantizing 8b parameter models.)

   1. [What is quantization anyways](Trasnformers.ipynb)
   2. [Tiny play expample of quatizing mock weights](quantization.ipynb)
   3. [Quantizing llama3.2-8b](quantizingopensourcemodels.ipynb)
      - fundamentally understanding of what quantization means
      - mechanics of quantization
      - quantizing llama3.2-8b from FB32 to 4-bit NF4 which reduces the size of model to just under 5Gb without sacrificing any noticable accuracy of model.
      - storing and using the quantized model
      - experiments quantizing and using below models
        1. Phi-3-mini-4k-instruct
        2. gemma-2-2b-it
        3. Qwen2-7B-Instruct
        4. Mixtral-8x7B-Instruct-v0.1

4. ## Fine Tuning opensource models.

   Below was done on a colab t4 instance.
   [steps 1 & 2](finetuning/data_prep.ipynb)

   1. Identifying and sourcing training data. I took a subset of amazon reviews from hugging face datasets.
   2. Scrubbing the training data (most important contributing factor to how well a model performs.) This took quite a while and cleaned data is now pushed back to [hugging face](https://huggingface.co/datasets/BharathkReddy/amazonreview-data-appliances_and_musicalinstruments)
   3. Experiments to [baseline predictions](finetuning/baselineModels.ipynb). Mean squared errors for below models compared.

      - Silly predictors (random predictor: 1966, mean predictor: 519)
      - Linear Regression (495): Generating features from unsctructured data was very gratifying and is quite an art. I left it early, but with enough investment of time, the MSE scores can drop significantly.
      - BagOfWords (337): frequency of words used as features for Linear regressor.
      - N-grams : since results from BagOfWords was so promicing, i couldnt resist doing a 2-gram and 3-gram with MSEs of 335 and 321. So clearly association of words do have predictive powers.
      - Word2Vec (370): Experiments from n-grams led me to think of using a bit more intelligent model which would learn associative power of words given surround words. I chose gensims implementation of word2vec for no particular reason. The vector matrix was used as features for a linear regressor.
      - Hypothesis testing (i wanted to see if Linear regressor was too simplistic to model the features from Word2Vec so i passed the word vector to few other models like Ridge , Random Forest and gradient Boost models to check both ends of spectrum - curtail the model to go mental with parameters. Results from all of these hovered around 503-509 so clearly model is not the problem - we need better word matrix!)
      - OpenAI 40-mini (1223): I wanted to check if OpenAI has already scraped this data and model somehow regurgiates all answers with 100% accuracy, my belief was misplaced as openAI couldnt beat more traditional methods here! but over much larger datasets LLMs would beat all known models on average.

   4. Experiments to understand the phenominon of `catastrophic forgetting`
