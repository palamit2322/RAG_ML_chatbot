from langchain_core.prompts import ChatPromptTemplate
system_prompt=(
    "Use the following context to answer the question"
    "If you don't find any context related to query just reply as no"
    "Don't give hallucinated answer of this"
    "Answer it in concise way"
    "\n\n"
    "Context:"
    "{context}"
   
)

prompt=ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        ("user","{question}")
    ]
)