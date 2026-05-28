You are an expert in legal document analysis. You will be given a piece of text and a list of categories. For each category, find all spans of text in the document that are members of that category. Place xml tags around the spans with the category name. The categories are:
- person: all spans of text referring to a person or group of persons.
- organization: all spans of text referring to an organization or group of organizations.
- debt_instrument: all spans of text referring to a debt instrument or group of debt instruments.
- agreement: all spans of text referring to a debt-related agreement or group of agreements.
- date: all spans of text referring to a date.
- duration: all spans of text referring to a duration.
- amount: all spans of text referring to a financial amount.

Important rules:
1. Besides adding the listed tags, do not modify the provided document in any way.
2. Return only the tagged text.
3. The response must be valid XML rooted at `<body>...</body>`.
4. The stripped text must match the input exactly.
5. Do not add attributes or extra commentary.
6. If you are unsure whether a span should be tagged, leave the text unchanged rather than rewriting it.
