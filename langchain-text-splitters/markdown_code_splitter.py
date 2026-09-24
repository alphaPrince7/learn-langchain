from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text = """
# 1. Main Document Title
## 1.1. Sub-heading Example

This is a paragraph featuring **bold text** and *italic text*. 
You can also use ~~strikethrough~~ to cross out words.

Here is an example of an [External Link](https://example.com).

### 1.1.1. Features & Code Samples
*   An unordered list item
*   Another bullet point with `inline_code_snippet`
    *   An indented sub-list item

1.  First item in an ordered list
2.  Second item in an ordered list

> This is a blockquote line for highlighting key information.

| Feature | Syntax | Status |
| :--- | :---: | ---: |
| Tables | Pipes | Supported |
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=200,
    chunk_overlap=0
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks[0])