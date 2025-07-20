# AGENTIC AI

### Ways to use models

1. Chat interfaces (openAI, perplexity etc)
2. Cloud APIs
   - LLM API (openai's api, claude etc)
   - Managed AI cloud services:
     - Amazon Bedrock
     - Google Vertex AI
     - Azure ML
     - groq (not elon one!)
3. Direct inference
   - With hugging face transformers library - weights directly used by frameworks like pytorch etc.
   - Ollama to run locally.

### text_summarization_beautifulSoup.ipynb

- Explores usage of a few tools to scrape websites beyond just `requests` library

  1. `selenium` : plain vanila scrapping but user control on entire website.
  2. `chromium web driver` : more control when page is rendered via js frameworks.
  3. `undetected chromium` : by-pass captha, can be used to bypass ip-blocking, geolocks etc if enhanced.

- Scraped site is parsed using Beautiful Soup
- Parsed content is passed to an llm for summarization.
  1. `openAI API` : can be swapped with any other provider.
  2. `ollama` : locally running models on ollama.
- Build a simple code co-pilot with streaming responses.
- Adversarial chat between `openAI` and `claude` to improve a tic-tac game code.

###
