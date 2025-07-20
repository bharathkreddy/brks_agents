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
